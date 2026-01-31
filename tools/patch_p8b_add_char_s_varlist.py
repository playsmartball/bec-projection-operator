import re
from pathlib import Path

F = Path('/mnt/e/CascadeProjects/bec-projection-operator-git/examples/dedalus_alfven_2d_nl_ivp.py')

def main():
    s = F.read_text(encoding='utf-8')
    if "var_list += [s_bx_l, s_bx_r, s_by_l, s_by_r]" in s:
        print('NO_CHANGES_ADD_CHAR_S')
        return
    # Insert only inside the char_bc branch after the tau1bx.. line
    pattern = (
        r"(if char_bc:\s*\n\s*var_list = \[.*?ctau_ry\]\s*\n\s*if eta > 0\.0:\s*\n\s*var_list \+= \[tau1bx, tau2bx, tau1by, tau2by\])"
    )
    repl = (r"\1\n            if (kappa_model == 'lowpass') and (kappa > 0.0) and (omega_c > 0.0):\n"
            r"                var_list += [s_bx_l, s_bx_r, s_by_l, s_by_r]")
    s2, n = re.subn(pattern, repl, s, count=1, flags=re.S)
    if n == 1:
        F.write_text(s2, encoding='utf-8')
        print('PATCH_OK_ADD_CHAR_S')
    else:
        print('NO_MATCH_ADD_CHAR_S')

if __name__ == '__main__':
    main()
