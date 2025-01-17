import numpy as np
import note_seq
import os
import tempfile
import functools
import copy
import tensorflow.compat.v1 as tf

from magenta.models.shared import events_rnn_model, events_rnn_graph
from magenta.models.drums_rnn import drums_rnn_model, drums_rnn_sequence_generator
from magenta.models.shared import sequence_generator_bundle
from magenta.common import state_util, beam_search
from note_seq import drums_encoder_decoder
from note_seq.protobuf import generator_pb2
from note_seq.protobuf import music_pb2

_DRUM_CLASS_REMAPPING = [
    0, 1, 2, 3, # kick, snare and hi-hats remain
    0,          # low tom to kick
    1,          # mid tom to snare
    1,          # high tom to snare
    3,          # crash to open hi-hat
    3,          # ride to open hi-hat
]

_DRUM_CLASS_REMAPPING_NO_HHO = [
    0, 1, 2,    # only kick, snare and closed hi-hat
    2,          # open hi-hat to closed
    0,          # the rest is the same
    1,
    1,
    2,
    2,
]

def _unary_idxs(n: int):
    idxs = []
    i = 0
    while n > 0:
        if n & 1 == 1:
            idxs.append(i)
        n = n >> 1
        i += 1
    return np.array(idxs)

def monophonizing_matrix(softmax_len):
    """
    Matrix which monophonizes softmaxes. Probabilities of
    events which include several hits at once are uniformly
    reassigned to each of the drums which were meant to be hit.
    """
    num_classes = np.log2(softmax_len)
    assert num_classes == int(num_classes)
    num_classes = int(num_classes)

    W = np.zeros((softmax_len, softmax_len))
    W[0, 0] = 1.0   # don't touch the silence
    for i in range(1, softmax_len):
        ixs = _unary_idxs(i)
        W[i, 2**ixs] = 1.0 / len(ixs)

    return W

def class_reduction_matrix(softmax_len, remapping=_DRUM_CLASS_REMAPPING):
    """
    Matrix which reduces number of drum classes involved.
    """
    W = np.zeros((softmax_len, softmax_len))
    W[0, 0] = 1.0   # don't touch the silence
    for from_cl, to_cl in enumerate(remapping):
        W[2**from_cl, 2**to_cl] = 1.0
    return W

def softmax_transform_matrix(softmax_len, remapping=_DRUM_CLASS_REMAPPING):
    M = monophonizing_matrix(softmax_len)
    R = class_reduction_matrix(softmax_len, remapping)
    return np.dot(M, R)

def drum_track_to_mono_classes(drum_track):
    cl = []
    enc = drums_encoder_decoder.MultiDrumOneHotEncoding()
    for e in drum_track:
        if len(e) > 1:
            raise ValueError('drum track is not mono')
        cl.append(enc.encode_event(e))
    return np.array(cl)

class MonoDrumsRnnModel(events_rnn_model.EventSequenceRnnModel):
    """
    Modified DrumsRNN for generating monophonic tracks and
    providing support for transcription.
    """
    
    def __init__(self, config, hho_included=True):
        super().__init__(config)
        self._hho_included = hho_included

    def generate_drum_track(self, num_steps, primer_drums, temperature=1.0,
                            beam_size=1, branch_factor=1, steps_per_iteration=1):
        return self._generate_events(num_steps, primer_drums, temperature,
                                     beam_size, branch_factor, steps_per_iteration)

    def drum_track_log_likelihood(self, drums):
        return self._evaluate_log_likelihood([drums])[0]

    def _modify_softmax_output(self):
        softmax = self._session.graph.get_collection('softmax')[0]
        self._session.graph.clear_collection('softmax')

        remapping = _DRUM_CLASS_REMAPPING if self._hho_included else _DRUM_CLASS_REMAPPING_NO_HHO
        transform_matrix = tf.convert_to_tensor(
            np.expand_dims(softmax_transform_matrix(softmax.shape[-1], remapping=remapping), 0), dtype=tf.float32)

        modified_softmax = tf.linalg.matmul(softmax, transform_matrix)
        self._session.graph.add_to_collection('softmax', modified_softmax)

    def _build_graph_for_generation(self):
        super(MonoDrumsRnnModel, self)._build_graph_for_generation()
        self._modify_softmax_output()

    def initialize_with_checkpoint_and_metagraph(self, checkpoint_filename, metagraph_filename):
        """
        Builds the TF graph with a checkpoint and metagraph.
        Copied/modified from `magenta.models.shared.model`
        """
        with tf.Graph().as_default():
            self._session = tf.Session()
            new_saver = tf.train.import_meta_graph(metagraph_filename)
            new_saver.restore(self._session, checkpoint_filename)
            self._modify_softmax_output()

    def modify_observation_probas(self, observations, temperature=1.0, beam_size=10,
                                  steps_per_iteration=1, **kwargs):
        """
        Given the (quantized) observation probabilities, perform a beam search
        with a language model and re-estimate those, returning the maximum
        likelihood sequence.
        """

        # We take the validity of the first observation as true, so we just prime the sequence
        # with the first decision of the classifier
        num_steps = len(observations)
        first_event_idx = np.argmax(observations[0])
        first_event_proba = observations[0][first_event_idx]
        primer_events = note_seq.DrumTrack(
            events=[self._config.encoder_decoder.class_index_to_event(first_event_idx, None)])

        event_sequences = [primer_events]
        inputs = self._config.encoder_decoder.get_inputs_batch(event_sequences, full_length=True)

        in_sts = self.get_initial_state()
        initial_states = state_util.unbatch(in_sts)

        initial_state = events_rnn_model.ModelState(inputs=inputs[0], rnn_state=initial_states[0],
                                                    control_events=None, control_state=None)

        generate_step_fn = functools.partial(
            self._generate_step_with_observations,
            observations=observations,
            temperature=temperature)

        events, _, loglik = beam_search(
            initial_sequence=event_sequences[0],
            initial_state=initial_state,
            generate_step_fn=generate_step_fn,
            num_steps=num_steps - len(primer_events),
            beam_size=beam_size,
            branch_factor=1,                           # our step function generates branches by itself
            steps_per_iteration=steps_per_iteration)

        tf.logging.info('Beam search yields sequence with log-likelihood: %f ',
                        loglik + np.log(first_event_proba))

        return events

    def _generate_step_with_observations(self, event_sequences, model_states, logliks,
                                         observations, temperature):

        # for simplicity, now just work only with batch size = 1
        batch_size = self._batch_size()
        assert batch_size == 1
        num_seqs = len(event_sequences)
        if num_seqs == 0:
            raise ValueError('No initial sequences provided')

        seq_len = len(event_sequences[0])
        if seq_len >= len(observations):
            raise ValueError(f'Length of the input sequence {seq_len} '\
                             f'is greater or equal than the length of observations {len(observations)}')

        # Extract inputs and RNN states from the model states.
        inputs = [model_state.inputs for model_state in model_states]
        initial_states = [model_state.rnn_state for model_state in model_states]
        result_states = []
        softmaxes = []

        for i in range(num_seqs):
            # Generate a single step for one batch of event sequences.
            batch_softmax, batch_final_state = self.eval_step(
                inputs[i:i+1],
                state_util.batch(initial_states[i:i+1], 1),
                temperature)

            # Adding observation probabilities to the equation
            batch_softmax *= observations[seq_len]
            result_states.append(state_util.unbatch(batch_final_state, 1)[0])
            softmaxes.append(batch_softmax[0])

        encoder_decoder = self._config.encoder_decoder
        final_event_sequences = []
        final_states = []
        final_logliks = []

        for i in range(num_seqs):
            sm = softmaxes[i][-1]
            result_state = result_states[i]
            nonzero_options = np.sum(sm > 0.0)
            assert nonzero_options <= 5    # just in case we encounter weird numeric stuff

            option_idxs = np.arange(len(sm))[sm > 0.0]
            seqs = []
            seq_states = []
            seq_logliks = []

            for opt in option_idxs:
                seq = copy.deepcopy(event_sequences[i])
                seq.append(encoder_decoder.class_index_to_event(opt, seq))
                seqs.append(seq)
                seq_states.append(copy.deepcopy(result_state))
                seq_logliks.append(logliks[i] + np.log(sm[opt]))

            final_event_sequences += seqs
            final_states += seq_states
            final_logliks += seq_logliks

        next_inputs = self._config.encoder_decoder.get_inputs_batch(final_event_sequences)
        model_states = [events_rnn_model.ModelState(inputs=inputs, rnn_state=final_state,
                                                    control_events=None, control_state=None)
                        for inputs, final_state in zip(next_inputs, final_states)]

        return final_event_sequences, model_states, final_logliks

    def eval_step(self, inputs, states, temperature=1.0):
        """
        Extends a batch of sequences by a single step each, taking into account
        given observations probability
        """
        assert len(inputs) == self._batch_size() and len(states) == self._batch_size()

        graph_inputs = self._session.graph.get_collection('inputs')[0]
        graph_initial_state = self._session.graph.get_collection('initial_state')
        graph_final_state = self._session.graph.get_collection('final_state')
        graph_softmax = self._session.graph.get_collection('softmax')[0]
        graph_temperature = self._session.graph.get_collection('temperature')[0]

        feed_dict = {
            graph_inputs: inputs,
            tuple(graph_initial_state): states,
            graph_temperature: temperature
        }
        final_state, softmax = self._session.run([graph_final_state, graph_softmax], feed_dict)
        return softmax, final_state

    def get_initial_state(self):
        graph_initial_state = self._session.graph.get_collection('initial_state')
        return self._session.run(graph_initial_state)

def read_bundle(bundle_file_path):
    bundle_file = os.path.expanduser(bundle_file_path)
    return sequence_generator_bundle.read_bundle_file(bundle_file)

def load_model_from_bundle(bundle_file_path, **kwargs):
    """
    Loads the model from the pre-trained bundle
    """
    bundle = read_bundle(bundle_file_path)
    config_id = bundle.generator_details.id
    config = drums_rnn_model.default_configs[config_id]

    # Extract checkpoint to tmpfile and load from it
    tempdir = tempfile.mkdtemp()
    ckpt_file = os.path.join(tempdir, f'model_{config_id}.ckpt')
    meta_file = os.path.join(tempdir, f'model_{config_id}.ckpt.meta')

    with open(ckpt_file, 'wb') as f:
        f.write(bundle.checkpoint_file[0])
    with open(meta_file, 'wb') as f:
        f.write(bundle.metagraph_file)

    model = MonoDrumsRnnModel(config, **kwargs)
    model.initialize_with_checkpoint_and_metagraph(ckpt_file, meta_file)

    return model

def load_generator_from_bundle(bundle_file_path):
    bundle = read_bundle(bundle_file_path)
    config_id = bundle.generator_details.id
    config = drums_rnn_model.default_configs[config_id]

    return drums_rnn_sequence_generator.DrumsRnnSequenceGenerator(
        model=MonoDrumsRnnModel(config),
        details=config.details,
        steps_per_quarter=config.steps_per_quarter,
        checkpoint=None,
        bundle=bundle)

def generate_track(generator, num_steps, qpm=120.0,
                   temperature=1.0, beam_size=1, branch_factor=1, steps_per_iteration=1):
    seconds_per_step = 60.0 / qpm / generator.steps_per_quarter
    total_seconds = num_steps * seconds_per_step

    generator_options = generator_pb2.GeneratorOptions()
    input_sequence = music_pb2.NoteSequence()
    input_sequence.tempos.add().qpm = qpm
    generate_section = generator_options.generate_sections.add(start_time=0, end_time=total_seconds)

    generator_options.args['temperature'].float_value = temperature
    generator_options.args['beam_size'].int_value = beam_size
    generator_options.args['branch_factor'].int_value = branch_factor
    generator_options.args['steps_per_iteration'].int_value = steps_per_iteration

    return generator.generate(input_sequence, generator_options)
