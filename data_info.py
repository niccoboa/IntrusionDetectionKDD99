# maps the attack types to the five classes (normal, dos, probe, r2l, u2r) in the KDD99 dataset.
def map_five_classes():
    return {
        'imap.': 'r2l',
        'named.': 'r2l',
        'land.': 'dos',
        'processtable.': 'dos',
        'back.': 'dos',
        'pod.': 'dos',
        'neptune.': 'dos',
        'buffer_overflow.': 'u2r',
        'teardrop.': 'dos',
        'apache2.': 'dos',
        'xsnoop.': 'r2l',
        'saint.': 'probe',
        'ftp_write.': 'r2l',
        'multihop.': 'r2l',
        'loadmodule.': 'u2r',
        'ipsweep.': 'probe',
        'nmap.': 'probe',
        'sqlattack.': 'r2l',
        'warezclient.': 'r2l',
        'snmpguess.': 'r2l',
        'satan.': 'probe',
        'portsweep.': 'probe',
        'xlock.': 'r2l',
        'mailbomb.': 'dos',
        'udpstorm.': 'dos',
        'worm.': 'u2r',
        'perl.': 'u2r',
        'snmpgetattack.': 'r2l',
        'normal.': 'normal',
        'smurf.': 'dos',
        'mscan.': 'probe',
        'warezmaster.': 'r2l',
        'sendmail.': 'r2l',
        'rootkit.': 'u2r',
        'guess_passwd.': 'r2l',
        'phf.': 'r2l',
        'spy.': 'u2r',
        'ps.': 'u2r',
        'xterm.': 'u2r',
        'httptunnel.': 'u2r'
    }

# Returns the 42 column names of the KDD99 dataset (including target label).
def get_column_names():
    columns = ['duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes', 'land', 'wrong_fragment',
               'urgent', 'hot', 'num_failed_logins', 'logged_in', 'num_compromised', 'root_shell', 'su_attempted',
               'num_root', 'num_file_creations', 'num_shells', 'num_access_files', 'num_outbound_cmds',
               'is_host_login', 'is_guest_login', 'count', 'srv_count', 'serror_rate', 'srv_serror_rate',
               'rerror_rate', 'srv_rerror_rate', 'same_srv_rate', 'diff_srv_rate', 'srv_diff_host_rate',
               'dst_host_count', 'dst_host_srv_count', 'dst_host_same_srv_rate', 'dst_host_diff_srv_rate',
               'dst_host_same_src_port_rate', 'dst_host_srv_diff_host_rate', 'dst_host_serror_rate',
               'dst_host_srv_serror_rate', 'dst_host_rerror_rate', 'dst_host_srv_rerror_rate', 'attack_type']
    return columns


def get_attack_types():
    return ['imap.', 'named.', 'land.', 'processtable.', 'back.', 'pod.', 'neptune.', 'buffer_overflow.', 'teardrop.',
            'apache2.', 'xsnoop.', 'saint.', 'ftp_write.', 'multihop.', 'loadmodule.', 'ipsweep.', 'nmap.', 'sqlattack.',
            'warezclient.', 'snmpguess.', 'satan.', 'portsweep.', 'xlock.', 'mailbomb.', 'udpstorm.', 'worm.', 'perl.',
            'snmpgetattack.', 'normal.', 'smurf.', 'mscan.', 'warezmaster.', 'sendmail.', 'rootkit.', 'guess_passwd.', 'phf.',
            'spy.', 'ps.', 'xterm.', 'httptunnel.']

def get_five_classes_attack_types():
    return ['normal.', 'dos.', 'probe.', 'r2l.', 'u2r.']