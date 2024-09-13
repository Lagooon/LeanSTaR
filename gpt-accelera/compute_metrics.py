import json
import glob

for setting in [
    'internLM2-7b-sample_mathlib_test1',
        ]:
    paths = [x for x in glob.glob('./output/%s/*' % setting)]
    #fs = [json.load(open(x)) for x in glob.glob('./output/%s/*' % setting)]
    print(paths)
    for pa in paths:
        fs = [json.load(open(x)) for x in glob.glob('%s/*.json' % pa)]
        n = 0
        ns = 0
        for f in fs:
            for result in f['results']:
                name = result['example']['full_name']

        # Extra helper theorem in the OpenAI code
                if 'sum_pairs' in name:
                    continue

                n += 1
                if result['success']:
                    ns += 1

        if n == 244 and ns / n > 0.36:
            print(pa, ns/n, ns, n, sep='\t')
