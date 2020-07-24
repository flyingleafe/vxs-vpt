import numpy as np
import note_seq
import os
import tempfile
import tensorflow.compat.v1 as tf

from magenta.models.shared import events_rnn_model, events_rnn_graph
from magenta.models.drums_rnn import drums_rnn_model, drums_rnn_sequence_generator
from magenta.models.shared import sequence_generator_bundle
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

class MonoDrumsRnnModel(events_rnn_model.EventSequenceRnnModel):
    """
    Modified DrumsRNN for generating monophonic tracks and 
    providing support for transcription.
    """

    def generate_drum_track(self, num_steps, primer_drums, temperature=1.0,
                            beam_size=1, branch_factor=1, steps_per_iteration=1):
        return self._generate_events(num_steps, primer_drums, temperature,
                                     beam_size, branch_factor, steps_per_iteration)

    def drum_track_log_likelihood(self, drums):
        return self._evaluate_log_likelihood([drums])[0]
    
    def _modify_softmax_output(self):
        softmax = self._session.graph.get_collection('softmax')[0]
        self._session.graph.clear_collection('softmax')
        
        transform_matrix = tf.convert_to_tensor(
            np.expand_dims(softmax_transform_matrix(softmax.shape[-1]), 0), dtype=tf.float32)
        
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

def read_bundle(bundle_file_path):
    bundle_file = os.path.expanduser(bundle_file_path)
    return sequence_generator_bundle.read_bundle_file(bundle_file)
        
def load_model_from_bundle(bundle_file_path):
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
        
    model = MonoDrumsRnnModel(config)
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