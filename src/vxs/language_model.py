import note_seq
import os
import tempfile
import tensorflow as tf

from magenta.models.shared import events_rnn_model
from magenta.models.drums_rnn import drums_rnn_model
from magenta.models.shared import sequence_generator_bundle
from note_seq.protobuf import generator_pb2

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

def load_model_from_bundle(bundle_file_path):
    """
    Loads the model from 
    """
    bundle_file = os.path.expanduser(bundle_file_path)
    bundle = sequence_generator_bundle.read_bundle_file(bundle_file)
    
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
    
