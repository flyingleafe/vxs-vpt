DEFAULT_SAMPLE_RATE = 44100

DEFAULT_ONSET_BUF_SIZE = 1024
DEFAULT_ONSET_HOP_SIZE = 512

DEFAULT_STFT_WINDOW = 4096
DEFAULT_STFT_HOP_SIZE = 512

EVENT_CLASSES = ['kd', 'sd', 'hhc', 'hho']

MIDI_PITCHES = {
    'kd': 36,
    'sd': 38,
    'hhc': 42,
    'hho': 46,
}

MIDI_PITCHES_SHORT = {
    'kd': 35,
    'sd': 27,
    'hhc': 44,
    'hho': 67,
}

SF2_AVP_INCLUDED_SAMPLES_FIXED = {
    1: ['sd_12', 'kd_23', 'kd_19', 'sd_25', 'hhc_29', 'hhc_1', 'hho_8', 'hho_15'],
    2: ['sd_13', 'kd_4', 'kd_23', 'sd_21', 'hhc_15', 'hhc_4', 'hho_23', 'hho_7'],
    3: ['sd_4', 'kd_8', 'kd_22', 'sd_23', 'hhc_23', 'hhc_16', 'hho_22', 'hho_0'],
    4: ['sd_3', 'kd_0', 'kd_17', 'sd_14', 'hhc_11', 'hhc_15', 'hho_8', 'hho_24'],
    5: ['sd_1', 'kd_4', 'kd_10', 'sd_0', 'hhc_0', 'hhc_19', 'hho_0', 'hho_1']
}

SF2_AVP_INCLUDED_SAMPLES_PERSONAL = {
    1: ['sd_0', 'kd_0', 'kd_2', 'sd_2', 'hhc_2', 'hhc_0', 'hho_2', 'hho_3'],
    2: ['sd_0', 'kd_3', 'kd_2', 'sd_2', 'hhc_2', 'hhc_0', 'hho_2', 'hho_0'],
    3: ['sd_0', 'kd_0', 'kd_2', 'sd_2', 'hhc_6', 'hhc_1', 'hho_2', 'hho_0'],
    4: ['sd_0', 'kd_0', 'kd_2', 'sd_2', 'hhc_2', 'hhc_0', 'hho_2', 'hho_0'],
    5: ['sd_1', 'kd_1', 'kd_0', 'sd_3', 'hhc_4', 'hhc_2', 'hho_0', 'hho_1'],
}

SF2_BEATBOXSET1_INCLUDED_SAMPLES = {
    'hex': ['s_2', 'k_7', 'k_0', 's_11', 'hc_12', 'hc_8', 'ho_5', 'ho_1'],
    'adidao': ['s_2', 'k_7', 'k_0', 's_11', 'hc_12', 'hc_8', 'ho_5', 'ho_1'],
    'daq': ['s_5', 'k_1', 'k_26', 's_8', 'hc_48', 'hc_4', 'ho_8', 'ho_5'],
    'bui': ['sk_25', 'k_3', 'k_35', 'sk_32', 'hc_3', 'hc_27', 'ho_4', 'ho_30'],
    'luckeymonkey': ['sk_14', 'k_3', 'k_12', 'sk_10', 'hc_0', 'hc_23', 'ho_0', 'ho_5'],
}

ANNOTATION_CLASSES = {
    'avp': ['kd', 'sd', 'hho', 'hhc'],
    'beatboxset1': ['k', 'hc', 'ho', 'sb', 'sk', 's']
}

BEATBOXSET1_CLASS_MAP = {
    'k': 'kd',
    'ho': 'hho',
    'hc': 'hhc',
    's': 'sd',
    'sb': 'sd',
    'sk': 'sd',
}
