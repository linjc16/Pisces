from ase.calculators.siesta.siesta import parse_siesta_version


def test_siesta_version():

    def test(txt, reference):
        buf = txt.encode('ascii')
        version = parse_siesta_version(buf)
        print(version)
        assert version == reference, (version, reference)

    # These reference version numbers have been found on the internet
    # by searching for "reinit: Reading from standard input".
    #
    # We don't support this one: "SIESTA 1.3    -- [Release] (30 Jul 2003)"
    test('Siesta Version: siesta-2.0.1', 'siesta-2.0.1')
    test('Siesta Version:  siesta-3.0-b', 'siesta-3.0-b')
    test('Siesta Version: siesta-3.0-rc2', 'siesta-3.0-rc2')
    test('Siesta Version:                                        siesta-3.1',
         'siesta-3.1')
    test('Siesta Version:                                        '
         'siesta-3.2-pl-5',
         'siesta-3.2-pl-5')
    test('Siesta Version: siesta-4.0--500', 'siesta-4.0--500')
    test('Siesta Version  : v4.0.2', 'v4.0.2')
    test('Siesta Version: siesta-4.1--736', 'siesta-4.1--736')
    test('  Version         : 5.0.0', '5.0.0')
