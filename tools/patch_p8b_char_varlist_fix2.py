from pathlib import Path

F = Path('/mnt/e/CascadeProjects/bec-projection-operator-git/examples/dedalus_alfven_2d_nl_ivp.py')

old = (
    "    if char_bc:\n"
    "        var_list = [vx, vy, bx, by, wx, wy, ctau_lx, ctau_rx, ctau_ly, ctau_ry]\n"
    "        if eta > 0.0:\n"
    "            var_list += [tau1bx, tau2bx, tau1by, tau2by]"
)
new = old + (
    "\n            if (kappa_model == 'lowpass') and (kappa > 0.0) and (omega_c > 0.0):\n"
    "                var_list += [s_bx_l, s_bx_r, s_by_l, s_by_r]"
)

def main():
    s = F.read_text(encoding='utf-8')
    if new in s:
        print('NO_CHANGES_CHAR_FIX2')
        return
    if old in s:
        s = s.replace(old, new)
        F.write_text(s, encoding='utf-8')
        print('PATCH_OK_CHAR_FIX2')
    else:
        print('NO_MATCH_CHAR_FIX2')

if __name__ == '__main__':
    main()
