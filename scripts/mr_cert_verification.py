import sys
import os
import re
import time
import platform
import subprocess
import json
from datetime import datetime
from multiprocessing import Pool, cpu_count, current_process

# FORCE PYTHON TO ALLOW UNLIMITED INTEGER STRINGS GLOBALLY
if hasattr(sys, 'set_int_max_str_digits'):
    sys.set_int_max_str_digits(0)

# Some Python/OS combinations re-execute this module's top-level code once per
# worker process when the Pool starts (e.g. under the "spawn" start method).
# Gate the startup banner to the main process only so it doesn't get reprinted.
_IS_MAIN_PROCESS = current_process().name == 'MainProcess'

try:
    import gmpy2
    USE_GMP = True
except ImportError:
    USE_GMP = False

RESERVED_CORES_DEFAULT = 2


def get_hardware_info():
    """Gathers verifier's CPU model, total RAM, and OS specifications across Linux, Windows, and Mac."""
    info = {'os': f"{platform.system()} {platform.release()} ({platform.architecture()})", 'cpu': "Unknown CPU", 'ram': "Unknown RAM"}

    try:
        if platform.system() == "Linux":
            with open("/proc/cpuinfo", "r") as f:
                for line in f:
                    if "model name" in line or "Model" in line:
                        info['cpu'] = line.split(":", 1)[1].strip()
                        break
        elif platform.system() == "Windows":
            info['cpu'] = platform.processor()
        elif platform.system() == "Darwin":
            os.environ['PATH'] = os.environ['PATH'] + os.pathsep + '/usr/sbin'
            info['cpu'] = subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"]).strip().decode()
    except Exception:
        pass

    try:
        if platform.system() == "Linux":
            with open("/proc/meminfo", "r") as f:
                for line in f:
                    if "MemTotal" in line:
                        kb = int(line.split()[1])
                        info['ram'] = f"{round(kb / (1024**2), 2)} GB"
                        break
        elif platform.system() == "Windows":
            import ctypes
            class MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong)
                ]
            stat = MEMORYSTATUSEX()
            stat.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat))
            info['ram'] = f"{round(stat.ullTotalPhys / (1024**3), 2)} GB"
        elif platform.system() == "Darwin":
            out = subprocess.check_output(["sysctl", "-n", "hw.memsize"]).strip().decode()
            info['ram'] = f"{round(int(out) / (1024**3), 2)} GB"
    except Exception:
        pass

    return info


def parse_certification_file(filename):
    expression = None
    bases = []
    with open(filename, 'r', encoding='utf-8') as f:
        content = f.read()
    expr_match = re.search(r"Expression Evaluated\s*:\s*(.+)", content)
    if expr_match:
        expression = expr_match.group(1).strip()
    base_matches = re.findall(r"Base \d+\.\d+:\s*(\d+)", content)
    for b in base_matches:
        bases.append(int(b))
    return expression, bases


def format_etr(seconds_remaining):
    """Converts seconds into a clean, human-readable format."""
    if seconds_remaining < 60:
        return f"{seconds_remaining:.1f} sec"
    minutes = int(seconds_remaining // 60)
    seconds = int(seconds_remaining % 60)
    if minutes < 60:
        return f"{minutes}m {seconds}s"
    hours = int(minutes // 60)
    minutes = int(minutes % 60)
    return f"{hours}h {minutes}m"


def get_available_core_count():
    """
    Number of logical cores this process could actually use.
    Prefers sched_getaffinity (respects taskset/cgroup CPU pinning) and falls
    back to the total system core count where that's unavailable (Windows/macOS).
    """
    if hasattr(os, "sched_getaffinity"):
        return len(os.sched_getaffinity(0))
    return cpu_count()


def choose_worker_count():
    total_available = get_available_core_count()
    suggested = max(1, total_available - RESERVED_CORES_DEFAULT)

    print(f"[!] {total_available} logical cores available to this process "
          f"(reserving {RESERVED_CORES_DEFAULT} by default for other work).")
    raw = input(f"    Worker processes to use [{suggested}]: ").strip()

    if raw == "":
        return suggested
    try:
        chosen = int(raw)
        if chosen < 1:
            print("    Invalid value, must be >= 1. Using default.")
            return suggested
        if chosen > total_available:
            print(f"    Warning: {chosen} exceeds {total_available} available cores - "
                  f"you may see thrashing/oversubscription.")
        return chosen
    except ValueError:
        print("    Couldn't parse that, using default.")
        return suggested


# ---- Worker: runs in a separate process. Must be top-level (picklable) and
# must re-import gmpy2 itself, since mpz objects don't cross process boundaries.
def _verify_base_worker(args):
    idx, a_int, n_str, d_str, s = args
    if USE_GMP:
        import gmpy2 as g
        n_gmp = g.mpz(n_str)
        d_gmp = g.mpz(d_str)
        n_minus_1 = n_gmp - 1
        x = g.powmod(g.mpz(a_int), d_gmp, n_gmp)
        if x == 1 or x == n_minus_1:
            return (idx, a_int, True)
        for _ in range(s - 1):
            x = g.powmod(x, 2, n_gmp)
            if x == n_minus_1:
                return (idx, a_int, True)
        return (idx, a_int, False)
    else:
        n = int(n_str)
        d = int(d_str)
        x = pow(a_int, d, n)
        if x == 1 or x == n - 1:
            return (idx, a_int, True)
        for _ in range(s - 1):
            x = pow(x, 2, n)
            if x == n - 1:
                return (idx, a_int, True)
        return (idx, a_int, False)


def load_recovery_ledger(recovery_file):
    """
    Returns the set of global base indices already confirmed verified in a
    prior (interrupted) run. Tolerates the older count-based ledger format
    from the serial script by treating it as "the first N indices are done".
    """
    if not os.path.exists(recovery_file):
        return set()
    try:
        with open(recovery_file, "r") as rf:
            data = json.load(rf)
        if "verified_indices" in data:
            return set(data["verified_indices"])
        if "verified_bases_count" in data:
            # Legacy serial-script ledger format.
            return set(range(data["verified_bases_count"]))
    except Exception as e:
        print(f"   [-] Warning reading recovery file (Starting fresh): {e}")
    return set()


def save_recovery_ledger(recovery_file, verified_indices):
    try:
        with open(recovery_file, "w") as wf:
            json.dump({"verified_indices": sorted(verified_indices)}, wf)
    except Exception:
        pass


def verify_certificate(cert_filename):
    print(f"=== Certificate Verification Audit Tool (Parallel) ===")
    if USE_GMP:
        print(f"[!] gmpy2 active (GMP v{gmpy2.mp_version()}): Verification math accelerated via GMP library.")
    else:
        print("[!] gmpy2 NOT found: Using fallback standard python evaluation blocks.")

    print(f"Reading target file: '{cert_filename}'...\n")
    try:
        expression, bases = parse_certification_file(cert_filename)
    except Exception as e:
        print(f"[-] Error reading or parsing the certificate file: {e}")
        return

    if not expression or not bases:
        print("[-] Error: Could not extract target data fields from certificate.")
        return

    print("Select Verification Mode:")
    print("  [1] Full Audit  (Verify all bases in the certificate - Maximum Certainty)")
    print("  [2] Quick Audit (Verify 10 randomly sampled bases - High Speed)")
    mode_choice = input("Select option (1 or 2, default is 1): > ").strip()

    audit_type = "FULL AUDIT"
    if mode_choice == "2":
        import random
        audit_type = "QUICK AUDIT (10 Randomly Sampled Bases)"
        if len(bases) > 10:
            bases = random.sample(bases, 10)

    n_workers = choose_worker_count()
    print(f"[!] Using {n_workers} worker process(es) for verification rounds.\n")

    print(f"[1/3] Parsing verification target expressions...")
    try:
        n = eval(expression)
    except Exception as e:
        print(f"[-] Error evaluating expression: {e}")
        return

    print(f"[2/3] Pre-calculating exponentiation factors...")
    d, s = n - 1, 0
    while d % 2 == 0:
        d //= 2
        s += 1

    n_str = str(n)
    d_str = str(d)

    print(f"[3/3] Commencing verification processing loop...")
    verifier_hw = get_hardware_info()
    verified = True
    failure_index = None
    failure_base = None

    fn_expr = expression.replace("**", "^").replace(" ", "")
    recovery_file = f"MR_{fn_expr}_verify_recovery.json"

    verified_indices = load_recovery_ledger(recovery_file)
    if verified_indices:
        print(f"   [+] Recovery Ledger Found: {len(verified_indices)} base(s) already confirmed. "
              f"Resuming remaining work...")

    k_total = len(bases)
    remaining_work = [(idx, bases[idx], n_str, d_str, s) for idx in range(k_total) if idx not in verified_indices]

    # Lower this process's (and its workers') scheduling priority so the OS
    # favors other running work on the machine when both want the CPU at once.
    if hasattr(os, "nice"):
        try:
            os.nice(10)
        except Exception:
            pass

    start_calc_time = time.perf_counter()
    completed_this_session = 0
    total_this_session = len(remaining_work)
    ledger_write_interval = 10  # base results between ledger flushes

    if total_this_session == 0:
        print("   [+] All bases already verified per recovery ledger - nothing left to do.")
    else:
        with Pool(processes=n_workers) as pool:
            for idx, a_int, passed in pool.imap_unordered(_verify_base_worker, remaining_work):
                completed_this_session += 1

                if not passed:
                    verified = False
                    failure_index = idx
                    failure_base = a_int
                    print(f"\n[!] VERIFICATION FAILED at base trace sequence element #{idx + 1} "
                          f"(base={a_int})!")
                    pool.terminate()
                    break

                verified_indices.add(idx)

                if completed_this_session % ledger_write_interval == 0:
                    save_recovery_ledger(recovery_file, verified_indices)

                if completed_this_session % 10 == 0 or completed_this_session == total_this_session:
                    elapsed = time.perf_counter() - start_calc_time
                    avg_time = elapsed / completed_this_session
                    remaining_iters = total_this_session - completed_this_session
                    print(f"   -> Progress: {len(verified_indices)}/{k_total} bases verified. "
                          f"[Ave. {avg_time:.3f} sec/base | ETR: {format_etr(remaining_iters * avg_time)}]")

        # Persist final state either way - lets a subsequent run resume past
        # everything confirmed good even if this run ultimately failed on one base.
        save_recovery_ledger(recovery_file, verified_indices)

    end_calc_time = time.perf_counter()
    total_calc_time = end_calc_time - start_calc_time
    avg_speed = total_calc_time / completed_this_session if completed_this_session > 0 else 0

    if os.path.exists(recovery_file) and verified:
        try:
            os.remove(recovery_file)
        except Exception:
            pass

    final_verdict = "VERIFIED SUCCESSFUL (PROBABLE PRIME)" if verified else "AUDIT FAILED (COMPOSITE OR TAMPERED)"
    out_filename = f"MR_{fn_expr}_verified.txt"

    print("\n" + "=" * 60)
    print(f" AUDIT VERDICT: {final_verdict}")
    print(f" Verification Type: {audit_type}")
    print("=" * 60)
    print(f"Core Calculation Time: {total_calc_time:.4f} seconds")
    print(f"Exporting independent verification log to: '{out_filename}'")
    print("=" * 60)

    try:
        with open(out_filename, "w", encoding="utf-8") as f:
            f.write("================================================================================\n")
            f.write("                       INDEPENDENT VERIFICATION LOG RECORD                      \n")
            f.write("================================================================================\n\n")
            f.write(f"Source Certificate File    : {cert_filename}\n")
            f.write(f"Target Number Expression   : {expression}\n")
            f.write(f"Verification Type Conducted: {audit_type}\n")
            f.write(f"Total Unique Bases Checked : {len(bases)}\n")
            f.write(f"Audit Status / Verdict     : {final_verdict}\n")
            if not verified and failure_index is not None:
                f.write(f"Failing Base (index / value): #{failure_index + 1} / {failure_base}\n")
            f.write("\n--- AUDIT ENVIRONMENT SIGNATURE (VERIFIER HARDWARE) ---\n")
            f.write(f"Verifier CPU Model         : {verifier_hw['cpu']}\n")
            f.write(f"Verifier RAM Capacity      : {verifier_hw['ram']}\n")
            f.write(f"Verifier Operating System  : {verifier_hw['os']}\n")
            f.write(f"Parallel Worker Processes  : {n_workers}\n\n")
            f.write("--- PERFORMANCE ANALYSIS ---\n")
            f.write(f"Total Audit Crunch Time    : {total_calc_time:.4f} seconds\n")
            f.write(f"Average Speed Per Base     : {avg_speed:.4f} seconds/base\n\n")
            f.write("================================================================================\n")
            f.write("      This file proves the target was independently audited and verified.     \n")
            f.write("================================================================================\n")
    except Exception as e:
        print(f"Error writing verification log file: {e}")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        verify_certificate(sys.argv[1])
    else:
        file_input = input("Enter the path to the certification text file:\n> ")
        verify_certificate(file_input.strip())
