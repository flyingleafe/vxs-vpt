import numpy as np
import note_seq

class BeamTreeNode:
    def __init__(self, step_idx, parent, prev_symbol, state, loglik):
        self.step_idx = step_idx
        self.parent = parent
        self.prev_symbol = prev_symbol
        self.state = state
        self.loglik = loglik

class BeamSearch:
    def __init__(self, transition_model, beam_size=10, steps_per_iteration=2, **kwargs):
        self.trans_model = transition_model
        self.beam_size = beam_size
        self.steps_per_iteration = steps_per_iteration
        self._reset()

    def _reset(self):
        init_node = BeamTreeNode(-1, None, None, None, 0.0)
        self.beam = [init_node]
        self.step_counter = 0

    def _encode_event_idx(self, event_idx, step_idx):
        """
        Encode an event without having to provide the whole input sequence.
        Lookbacks are not supported.
        """
        enc_dec = self.trans_model._config.encoder_decoder
        input_ = np.zeros(enc_dec.input_size)
        input_[event_idx] = 1.0

        offset = enc_dec._one_hot_encoding.num_classes
        n = step_idx + 1
        for i in range(enc_dec._binary_counter_bits):
            input_[offset] = 1.0 if (n // 2 ** i) % 2 else -1.0
            offset += 1

        return np.expand_dims(input_, 0)

    def run(self, observations, no_reset=False, temperature=1.0, **kwargs):
        if not no_reset:
            self._reset()

        for i in range(len(observations)):
            self.step(observations[i], temperature=temperature)
            if self.step_counter % self.steps_per_iteration == 0:
                self.purge()

        best_node = max(self.beam, key=lambda n: n.loglik)
        # print('Beam search yields sequence with likelihood {}'.format(best_node.loglik))

        events = []
        while best_node.parent is not None:
            events.append(best_node.prev_symbol)
            best_node = best_node.parent

        events.reverse()
        assert len(events) == len(observations)

        enc_dec = self.trans_model._config.encoder_decoder
        drums = note_seq.DrumTrack(events=[
            enc_dec.class_index_to_event(idx, None)
            for idx in events
        ])

        return drums

    def step(self, observ_probas, temperature=1.0):
        next_beam = []
        for node in self.beam:
            if node.parent is not None:
                init_states = node.state
                inputs = [self._encode_event_idx(node.prev_symbol, node.step_idx)]
                softmax, state = self.trans_model.eval_step(
                    inputs, init_states, temperature=temperature)
                probas = softmax[0][-1] * observ_probas
            else:
                probas = observ_probas
                state = self.trans_model.get_initial_state()

            nz_idxs, = np.where(probas > 0)
            for nx_ix in nz_idxs:
                new_loglik = node.loglik + np.log(probas[nx_ix])
                new_node = BeamTreeNode(self.step_counter, node, nx_ix, state, new_loglik)
                next_beam.append(new_node)
        self.beam = next_beam
        self.step_counter += 1

    def purge(self):
        self.beam.sort(key=lambda n: -n.loglik)
        self.beam = self.beam[:self.beam_size]


def beam_search_decode(trans_model, observations, **kwargs):
    bms = BeamSearch(trans_model, **kwargs)
    return bms.run(observations, **kwargs)
