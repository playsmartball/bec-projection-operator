import sys
import re
from pathlib import Path

F = Path('/mnt/e/CascadeProjects/bec-projection-operator-git/examples/dedalus_alfven_2d_nl_ivp.py')

def main():
    s = F.read_text(encoding='utf-8')
    changed = False

    # 1) Extend function signature
    sig_old = "kappa: float = 0.0, bc: str = 'characteristic'):"
    sig_new = "kappa: float = 0.0, bc: str = 'characteristic', kappa_model: str = 'constant', omega_c: float = 0.0):"
    if sig_old in s and sig_new not in s:
        s = s.replace(sig_old, sig_new)
        changed = True

    # 2) Insert auxiliary boundary states after ctau_ry
    if "s_bx_l = dist.Field(name='s_bx_l'" not in s:
        anchor = "    ctau_ry = dist.Field(name='ctau_ry', bases=(xbasis,))"
        if anchor in s:
            ins = anchor + ("\n"
                "    s_bx_l = dist.Field(name='s_bx_l', bases=(xbasis,))\n"
                "    s_bx_r = dist.Field(name='s_bx_r', bases=(xbasis,))\n"
                "    s_by_l = dist.Field(name='s_by_l', bases=(xbasis,))\n"
                "    s_by_r = dist.Field(name='s_by_r', bases=(xbasis,))")
            s = s.replace(anchor, ins)
            changed = True

    # 3) Conditionally add states to var_list (char_bc & eta>0)
    if "var_list += [s_bx_l, s_bx_r, s_by_l, s_by_r]" not in s:
        anchor2 = "            var_list += [tau1bx, tau2bx, tau1by, tau2by]"
        if anchor2 in s:
            ins2 = anchor2 + ("\n"
                "            if (kappa_model == 'lowpass') and (kappa > 0.0) and (omega_c > 0.0):\n"
                "                var_list += [s_bx_l, s_bx_r, s_by_l, s_by_r]")
            s = s.replace(anchor2, ins2)
            changed = True

    # 4) BC logic: split constant vs lowpass
    # Narrow replacement of the first occurrence only
    if "if (kappa > 0.0) and (kappa_model == 'constant'):" not in s:
        s = s.replace("if kappa > 0.0:", "if (kappa > 0.0) and (kappa_model == 'constant'):", 1)
        changed = True
    anchor3 = "                problem.add_equation(\" d_z(by)(z='right') + kappa*by(z='right') = 0\")"
    if anchor3 in s and "dt(s_bx_l)" not in s:
        elif_block = anchor3 + ("\n"
            "            elif (kappa > 0.0) and (kappa_model == 'lowpass') and (omega_c > 0.0):\n"
            "                problem.add_equation(\"dt(s_bx_l) - omega_c*kappa*bx(z='left') + omega_c*s_bx_l = 0\")\n"
            "                problem.add_equation(\"dt(s_by_l) - omega_c*kappa*by(z='left') + omega_c*s_by_l = 0\")\n"
            "                problem.add_equation(\"dt(s_bx_r) - omega_c*kappa*bx(z='right') + omega_c*s_bx_r = 0\")\n"
            "                problem.add_equation(\"dt(s_by_r) - omega_c*kappa*by(z='right') + omega_c*s_by_r = 0\")\n"
            "                problem.add_equation(\"-d_z(bx)(z='left') + s_bx_l = 0\")\n"
            "                problem.add_equation(\" d_z(bx)(z='right') + s_bx_r = 0\")\n"
            "                problem.add_equation(\"-d_z(by)(z='left') + s_by_l = 0\")\n"
            "                problem.add_equation(\" d_z(by)(z='right') + s_by_r = 0\")")
        s = s.replace(anchor3, elif_block)
        changed = True

    # 5) CLI args
    if "--kappa_model" not in s:
        anchor4 = "    p.add_argument(\"--kappa\", type=float, default=0.0)"
        if anchor4 in s:
            ins4 = anchor4 + ("\n"
                "    p.add_argument(\"--kappa_model\", type=str, choices=['constant','lowpass'], default='constant')\n"
                "    p.add_argument(\"--omega_c\", type=float, default=0.0)")
            s = s.replace(anchor4, ins4)
            changed = True

    # 6) Plumb args to run_nl_ivp
    if "kappa_model=args.kappa_model" not in s:
        call_anchor = "kappa=args.kappa, bc=args.bc)"
        call_replace = "kappa=args.kappa, bc=args.bc, kappa_model=args.kappa_model, omega_c=args.omega_c)"
        if call_anchor in s:
            s = s.replace(call_anchor, call_replace)
            changed = True

    if changed:
        F.write_text(s, encoding='utf-8')
        print('PATCH_OK')
    else:
        print('NO_CHANGES')

if __name__ == '__main__':
    main()
