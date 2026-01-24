import os

def parse_step2(path):
    if not os.path.exists(path):
        return None
    rows = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip().startswith('form') or line.strip().startswith('sigma8_base'):
                continue
            parts = line.strip().split('\t')
            if len(parts) >= 6:
                form, n, kc, s8, delta, cls = parts[:6]
                rows.append({
                    'form': form,
                    'n': n,
                    'kc': kc,
                    'sigma8': s8,
                    'delta': delta,
                    'class': cls
                })
    return rows


def parse_step3(path):
    if not os.path.exists(path):
        return None
    info = {'aborted': None, 'delta': None}
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            s = line.strip()
            if s.startswith('delta = '):
                info['delta'] = s.split('=')[1].strip()
            if s.startswith('ABORT:'):
                info['aborted'] = True
            if s.startswith('OK:'):
                info['aborted'] = False
    return info


def parse_step4(path):
    if not os.path.exists(path):
        return None
    overall = None
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            s = line.strip()
            if s.startswith('OVERALL:'):
                overall = s.split(':', 1)[1].strip()
                break
    return overall


def decide(step2_rows, step3_info, step4_overall):
    # Default: cannot decide without Step 2
    if not step2_rows:
        return 'σ₈ not explainable by projection'

    any_pass = any(r.get('class', '').upper() == 'PASS' for r in step2_rows)
    any_over = any(r.get('class', '').upper() == 'OVER-SUPPRESSED' for r in step2_rows)

    if not any_pass:
        return 'σ₈ not explainable by projection'

    # Step 3/4 required for a full "explained" classification
    if step3_info is None or step4_overall is None:
        return 'σ₈ partially projection-driven'

    if step3_info.get('aborted'):
        return 'σ₈ partially projection-driven'

    if step4_overall != 'PASS':
        return 'σ₈ partially projection-driven'

    return 'σ₈ explained as projection artifact'


def main():
    os.makedirs(os.path.join('output', 'summaries'), exist_ok=True)
    p2 = os.path.join('output', 'summaries', 'phase16b_sigma8_null_test.txt')
    p3 = os.path.join('output', 'summaries', 'phase16c_sigma8_operator.txt')
    p4 = os.path.join('output', 'summaries', 'phase16d_cross_checks.txt')

    step2_rows = parse_step2(p2)
    step3_info = parse_step3(p3)
    step4_overall = parse_step4(p4)

    decision = decide(step2_rows, step3_info, step4_overall)

    out = os.path.join('output', 'summaries', 'phase16e_decision.txt')
    with open(out, 'w', encoding='utf-8') as f:
        f.write(decision + '\n')

    print(f'Wrote {out}: {decision}')


if __name__ == '__main__':
    main()
