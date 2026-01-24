#!/usr/bin/env python3
import argparse
import csv as _csv
import json
import os
import re
import time

def parse_metrics_line(line):
    i = line.find("METRICS_JSON=")
    if i < 0:
        return None
    s = line[i + len("METRICS_JSON="):].strip()
    try:
        return json.loads(s)
    except Exception:
        return None

def parse_tags_from_csv_path(path):
    fname = os.path.basename(path)
    m = re.search(r"p8b_k([^_]+)_oc(\d+)_([a-zA-Z]+)\.csv$", fname)
    if not m:
        return None
    kappa_s = m.group(1)
    oc_digits = m.group(2)
    forcing = m.group(3)
    try:
        kappa = float(kappa_s)
    except Exception:
        try:
            kappa = float(kappa_s.replace('p', 'e'))
        except Exception:
            kappa = float('nan')
    try:
        oc_frac = float(int(oc_digits)) / 1000.0
    except Exception:
        oc_frac = float('nan')
    return dict(kappa=kappa, oc_frac=oc_frac, forcing=forcing)

def read_omega0(path):
    try:
        with open(path, 'r') as f:
            s = f.read().strip()
            return float(s)
    except Exception:
        return float('nan')

def load_existing(summary_path):
    rows = {}
    if not os.path.exists(summary_path):
        return rows
    try:
        with open(summary_path, newline='') as f:
            r = _csv.DictReader(f)
            for row in r:
                rows[row.get('csv', '')] = row
    except Exception:
        pass
    return rows

def write_summary(summary_path, rows_list):
    cols = [
        'kappa','oc_frac','omega_c','forcing',
        'deltaE_over_E','deltaE_over_E_weak','W_tau_over_E0',
        'P_eta_total_final','P_eta_total_slope_lastk',
        'peta_fft_fraction_above_omega_c','dominance','csv'
    ]
    tmp = summary_path + '.tmp'
    with open(tmp, 'w', newline='') as f:
        w = _csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for row in rows_list:
            w.writerow({k: row.get(k, '') for k in cols})
    os.replace(tmp, summary_path)

def process_metrics(mj, omega0):
    csvp = mj.get('csv', '')
    tags = parse_tags_from_csv_path(csvp)
    if not tags:
        return None
    kappa = tags['kappa']
    oc_frac = tags['oc_frac']
    forcing = tags['forcing']
    omega_c = oc_frac * omega0 if omega0 == omega0 else float('nan')
    row = {
        'kappa': kappa,
        'oc_frac': oc_frac,
        'omega_c': omega_c,
        'forcing': forcing,
        'deltaE_over_E': mj.get('deltaE_over_E',''),
        'deltaE_over_E_weak': mj.get('deltaE_over_E_weak',''),
        'W_tau_over_E0': mj.get('W_tau_over_E0',''),
        'P_eta_total_final': mj.get('P_eta_total_final',''),
        'P_eta_total_slope_lastk': mj.get('P_eta_total_slope_lastk',''),
        'peta_fft_fraction_above_omega_c': mj.get('peta_fft_fraction_above_omega_c',''),
        'dominance': mj.get('dominance',''),
        'csv': csvp,
    }
    return row

def initial_pass(log_path, omega0, summary_path):
    acc = load_existing(summary_path)
    try:
        with open(log_path, 'r', errors='ignore') as f:
            for line in f:
                mj = parse_metrics_line(line)
                if mj:
                    row = process_metrics(mj, omega0)
                    if row:
                        acc[row['csv']] = row
    except FileNotFoundError:
        pass
    write_summary(summary_path, list(acc.values()))
    return acc

def tail_follow(log_path, omega0, summary_path, acc):
    try:
        with open(log_path, 'r', errors='ignore') as f:
            f.seek(0, os.SEEK_END)
            while True:
                line = f.readline()
                if not line:
                    time.sleep(1.0)
                    continue
                mj = parse_metrics_line(line)
                if mj:
                    row = process_metrics(mj, omega0)
                    if row:
                        acc[row['csv']] = row
                        write_summary(summary_path, list(acc.values()))
                        print(f"UPDATED {row['csv']} kappa={row['kappa']} oc={row['oc_frac']} forcing={row['forcing']} frac_above={row['peta_fft_fraction_above_omega_c']}")
    except KeyboardInterrupt:
        return

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--log', default='/mnt/e/CascadeProjects/bec-projection-operator-git/analysis/phase8_runs/phase8b_sweep.log')
    ap.add_argument('--summary_csv', default='/mnt/e/CascadeProjects/bec-projection-operator-git/analysis/phase8_runs/p8b_summary.csv')
    ap.add_argument('--omega0_file', default='/mnt/e/CascadeProjects/bec-projection-operator-git/analysis/phase8_runs/p8b_omega0.txt')
    args = ap.parse_args()
    omega0 = read_omega0(args.omega0_file)
    acc = initial_pass(args.log, omega0, args.summary_csv)
    tail_follow(args.log, omega0, args.summary_csv, acc)

if __name__ == '__main__':
    main()
