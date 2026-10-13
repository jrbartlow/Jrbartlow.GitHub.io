import sys
import os
import math
import platform
import time
import json
import ast
from datetime import datetime
import secrets
from multiprocessing import Pool, cpu_count, current_process

if hasattr(sys, 'set_int_max_str_digits'):
    sys.set_int_max_str_digits(0)

_IS_MAIN_PROCESS = current_process().name == 'MainProcess'

try:
    import gmpy2
    USE_GMP = True
    if _IS_MAIN_PROCESS:
        print(f"[SUCCESS] gmpy2 loaded successfully! C-GMP Engine Active (v{gmpy2.mp_version()}).")
except ImportError as e:
    USE_GMP = False
    if _IS_MAIN_PROCESS:
        print(f"[WARNING] gmpy2 NOT FOUND: {e}")

FIRST_200_PRIMES = [
    2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97,
    101, 103, 107, 109, 113, 127, 131, 137, 139, 149, 151, 157, 163, 167, 173, 179, 181, 191, 193, 197, 199,
    211, 223, 227, 229, 233, 239, 241, 251, 257, 263, 269, 271, 277, 281, 283, 293, 307, 311, 313, 317, 331,
    337, 347, 349, 353, 359, 367, 373, 379, 383, 389, 397, 401, 409, 419, 421, 431, 433, 439, 443, 449, 457,
    461, 463, 467, 479, 487, 491, 499, 503, 509, 521, 523, 541, 547, 557, 563, 569, 571, 577, 587, 593, 599,
    601, 607, 613, 617, 619, 631, 641, 643, 647, 653, 659, 661, 673, 677, 683, 691, 701, 709, 719, 727, 733,
    739, 743, 751, 757, 761, 769, 773, 787, 797, 809, 811, 821, 823, 827, 829, 839, 853, 857, 859, 863, 877,
    881, 883, 887, 907, 911, 919, 929, 937, 941, 947, 953, 967, 971, 977, 983, 991, 997, 1009, 1013, 1019,
    1021, 1031, 1033, 1039, 1049, 1051, 1061, 1063, 1069, 1087, 1091, 1093, 1097, 1103, 1109, 1117, 1123,
    1129, 1151, 1153, 1163, 1171, 1181, 1187, 1193, 1201, 1213, 1217, 1223
]

K1_DEFAULT = 100
K2_DEFAULT = 100
RESERVED_CORES_DEFAULT = 2
BACKUP_FILE = "mr_checkpoint.json"


def safe_eval_expr(expr_str):
    """Safely evaluates basic arithmetic expressions without arbitrary code execution risk."""
    tree = ast.parse(expr_str, mode='eval')
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Expression, ast.BinOp, ast.UnaryOp, ast.Constant,
                                ast.Add, ast.Sub, ast.Mult, ast.Pow, ast.Div, ast.USub, ast.UAdd)):
            raise ValueError(f"Unsupported math operation or syntax in expression: {ast.dump(node)}")
    return eval(compile(tree, filename='', mode='eval'))


def get_next_prime(current_p):
    """Fallback prime generator if k1 > 200."""
    p = current_p + 1
    while True:
        if gmpy2.is_prime(p):
            return p
        p += 1


def get_available_core_count():
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


def get_hardware_info():
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
    except Exception:
        pass
    return info


def calculate_probabilities(k_total, k1, k2, prime1, prime2):
    def to_sci(k):
        log_p = -k * math.log10(4)
        return f"{10 ** (log_p - math.floor(log_p)):.4f} x 10^{math.floor(log_p)}"
    return (to_sci(k1) if prime1 else "1.0000", to_sci(k2) if prime2 else "1.0000",
            to_sci(k_total) if (prime1 and prime2) else "1.0000")


def format_etr(seconds_remaining):
    if seconds_remaining < 60:
        return f"{seconds_remaining:.1f} sec"
    minutes = int(seconds_remaining // 60)
    seconds = int(seconds_remaining % 60)
    if minutes < 60:
        return f"{minutes}m {seconds}s"
    hours = int(minutes // 60)
    minutes = int(minutes % 60)
    return f"{hours}h {minutes}m"


def save_checkpoint(state):
    """Saves process state to a JSON file."""
    with open(BACKUP_FILE, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=4)


def load_checkpoint():
    """Loads state if a valid checkpoint exists."""
    if os.path.exists(BACKUP_FILE):
        try:
            with open(BACKUP_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            print(f"[!] Checkpoint exists but could not be read: {e}")
    return None


def clear_checkpoint():
    """Deletes checkpoint on successful run completion."""
    if os.path.exists(BACKUP_FILE):
        try:
            os.remove(BACKUP_FILE)
        except Exception:
            pass


def _mr_round_worker(args):
    a_int, n_str, d_str, s = args
    import gmpy2 as g
    n_gmp = g.mpz(n_str)
    d_gmp = g.mpz(d_str)
    n_minus_1 = n_gmp - 1
    a_gmp = g.mpz(a_int)

    x = g.powmod(a_gmp, d_gmp, n_gmp)
    if x == 1 or x == n_minus_1:
        return (a_int, True)
    for _ in range(s - 1):
        x = g.powmod(x, 2, n_gmp)
        if x == n_minus_1:
            return (a_int, True)
    return (a_int, False)


def generate_bases(stage_mode, k_iterations, n, global_used_set, existing_bases=None):
    bases = list(existing_bases) if existing_bases else []
    for b in bases:
        global_used_set.add(b)

    last_prime = FIRST_200_PRIMES[-1] if len(bases) >= len(FIRST_200_PRIMES) else 2

    while len(bases) < k_iterations:
        if stage_mode == "primes":
            idx = len(bases)
            if idx < len(FIRST_200_PRIMES):
                a_int = FIRST_200_PRIMES[idx]
            else:
                last_prime = get_next_prime(last_prime)
                a_int = last_prime
        else:
            a_int = secrets.randbits(64)

        if a_int in global_used_set or a_int >= n:
            continue
        global_used_set.add(a_int)
        bases.append(a_int)
    return bases


def run_single_mr_test_parallel(n, stage_mode, k_iterations, d, s, stage_name,
                                 global_used_set, pool, n_str, d_str, checkpoint_data,
                                 expr_clean, k1, k2):
    resumed_stage = checkpoint_data is not None and checkpoint_data.get("current_stage") == stage_name
    stage_key = "stage1_bases" if stage_name == "Stage 1" else "stage2_bases"
    already_completed_bases = checkpoint_data.get(stage_key, []) if resumed_stage else []
    already_passed_count = len(already_completed_bases)

    bases = generate_bases(stage_mode, k_iterations, n, global_used_set, existing_bases=already_completed_bases)

    start_stage_time = time.perf_counter()
    completed = already_passed_count
    all_passed = True

    remaining_bases = bases[already_passed_count:]
    work_items = [(a, n_str, d_str, s) for a in remaining_bases]
    completed_bases = list(already_completed_bases)

    if completed > 0:
        print(f"   [+] Resuming {stage_name} from checkpoint at {completed}/{k_iterations} iterations.")

    for a_int, passed in pool.imap_unordered(_mr_round_worker, work_items):
        completed += 1
        completed_bases.append(a_int)

        if not passed:
            all_passed = False

        if completed % 10 == 0 or completed == k_iterations or not all_passed:
            elapsed = time.perf_counter() - start_stage_time
            new_iters = completed - already_passed_count
            avg_time = (elapsed / new_iters) if new_iters > 0 else 0
            remaining_iters = k_iterations - completed
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            print(f"   -> [{timestamp}] {stage_name}: {completed}/{k_iterations} iterations complete. "
                  f"[Avg. {avg_time:.4f} sec/iter | ETR: {format_etr(remaining_iters * avg_time)}]")

            state = {
                "expr_clean": expr_clean,
                "current_stage": stage_name,
                "k1": k1,
                "k2": k2,
                "stage1_complete": checkpoint_data.get("stage1_complete", False) if checkpoint_data else False,
                "stage1_passed": checkpoint_data.get("stage1_passed", True) if checkpoint_data else True,
                "stage1_bases": checkpoint_data.get("stage1_bases", []) if checkpoint_data else [],
                "start1": checkpoint_data.get("start1") if checkpoint_data else None,
                "end1": checkpoint_data.get("end1") if checkpoint_data else None,
                "stage2_complete": False,
                "stage2_passed": all_passed,
                "stage2_bases": completed_bases if stage_name == "Stage 2" else [],
                "start2": checkpoint_data.get("start2") if checkpoint_data else None,
                "end2": checkpoint_data.get("end2") if checkpoint_data else None
            }
            if stage_name == "Stage 1":
                state["stage1_bases"] = completed_bases
                state["stage1_passed"] = all_passed

            save_checkpoint(state)

        if not all_passed:
            break

    return all_passed, bases


def main():
    print("=== Certified Miller-Rabin Dual-Stage Primality Verifier (Parallel) ===")
    if USE_GMP:
        print(f"[!] gmpy2 active (GMP v{gmpy2.mp_version()}): Fast C-GMP Core Enabled.")
    else:
        print("[!] WARNING: Running without gmpy2 - this script requires it for the worker processes.")
        sys.exit(1)

    checkpoint = load_checkpoint()

    if checkpoint:
        print(f"\n[!] Found saved checkpoint for expression: '{checkpoint['expr_clean']}'")
        res = input("    Resume previous session? [Y/n]: ").strip().lower()
        if res not in ('', 'y', 'yes'):
            checkpoint = None
            clear_checkpoint()

    if checkpoint:
        expr_clean = checkpoint['expr_clean']
        print(f"Resuming computation for expression: {expr_clean}")
    else:
        expr_input = input("Enter expression to test (e.g., 7**20176 - 2):\n> ")
        expr_clean = expr_input.strip()

    n_workers = choose_worker_count()
    print(f"[!] Using {n_workers} worker process(es) for modexp rounds.\n")

    print("\nEvaluating expression...")
    n = safe_eval_expr(expr_clean)
    k1 = checkpoint['k1'] if checkpoint else K1_DEFAULT
    k2 = checkpoint['k2'] if checkpoint else K2_DEFAULT
    k_total = k1 + k2

    fn_expr = expr_clean.replace("**", "^").replace(" ", "")
    digit_length = int(gmpy2.log10(gmpy2.mpz(n))) + 1

    print(f"\n[1/4] Gathering system info...")
    hw = get_hardware_info()

    print("[2/4] Factoring exponent sequence...")
    d, s = n - 1, 0
    while d % 2 == 0:
        d //= 2
        s += 1

    n_str = str(n)
    d_str = str(d)
    global_used_set = set()

    if hasattr(os, "nice"):
        try:
            os.nice(10)
        except Exception:
            pass

    start1_str = checkpoint.get("start1") if checkpoint else None
    end1_str = checkpoint.get("end1") if checkpoint else None
    start2_str = checkpoint.get("start2") if checkpoint else None
    end2_str = checkpoint.get("end2") if checkpoint else None

    with Pool(processes=n_workers) as pool:
        # Stage 1 Execution
        if checkpoint and checkpoint.get("stage1_complete"):
            print(f"[3/4] Skipping Stage 1 (already completed in checkpoint).")
            prime1 = checkpoint["stage1_passed"]
            bases1 = checkpoint["stage1_bases"]
        else:
            print(f"[3/4] Launching Test Run 1 (First {k1} consecutive primes) across {n_workers} workers...")
            start1_dt = datetime.now()
            start1_str = start1_dt.isoformat()
            prime1, bases1 = run_single_mr_test_parallel(
                n, "primes", k1, d, s, "Stage 1", global_used_set, pool, n_str, d_str,
                checkpoint, expr_clean, k1, k2)
            end1_dt = datetime.now()
            end1_str = end1_dt.isoformat()

            checkpoint_state = {
                "expr_clean": expr_clean,
                "current_stage": "Stage 1",
                "k1": k1, "k2": k2,
                "stage1_complete": True,
                "stage1_passed": prime1,
                "stage1_bases": bases1,
                "start1": start1_str, "end1": end1_str,
                "stage2_complete": False,
                "stage2_passed": True,
                "stage2_bases": [],
                "start2": None, "end2": None
            }
            save_checkpoint(checkpoint_state)
            checkpoint = checkpoint_state

        # Stage 2 Execution
        if not prime1:
            final_verdict = "DEFINITIVELY COMPOSITE"
            prime2, bases2 = False, []
            start2_str = end2_str = datetime.now().isoformat()
            print(f"\n[RESULT] Stage 1 found a witness proving compositeness - skipping Stage 2.")
        else:
            if checkpoint and checkpoint.get("stage2_complete"):
                print(f"[4/4] Skipping Stage 2 (already completed in checkpoint).")
                prime2 = checkpoint["stage2_passed"]
                bases2 = checkpoint["stage2_bases"]
            else:
                print(f"[4/4] Launching Test Run 2 ({k2} random 64-bit bases) across {n_workers} workers...")
                start2_dt = datetime.now()
                start2_str = start2_dt.isoformat()
                prime2, bases2 = run_single_mr_test_parallel(
                    n, "random", k2, d, s, "Stage 2", global_used_set, pool, n_str, d_str,
                    checkpoint, expr_clean, k1, k2)
                end2_dt = datetime.now()
                end2_str = end2_dt.isoformat()

            final_verdict = "PROBABLE PRIME (PRP)" if prime2 else "DEFINITIVELY COMPOSITE"

    # Convert timestamps safely back to datetime for formatting
    start1 = datetime.fromisoformat(start1_str) if start1_str else datetime.now()
    end1 = datetime.fromisoformat(end1_str) if end1_str else datetime.now()
    start2 = datetime.fromisoformat(start2_str) if start2_str else datetime.now()
    end2 = datetime.fromisoformat(end2_str) if end2_str else datetime.now()

    print(f"\n{'=' * 60}")
    print(f"  VERDICT: {final_verdict}")
    print(f"{'=' * 60}\n")

    p1_str, p2_str, comb_str = calculate_probabilities(k_total, k1, k2, prime1, prime2)

    cert_filename = f"MR_{fn_expr}_certification.txt"
    with open(cert_filename, "w", encoding="utf-8") as f:
        f.write("================================================================================\n")
        f.write("               PRIMALITY TEST VERIFICATION & CERTIFICATE RECORD                 \n")
        f.write("================================================================================\n\n")
        f.write("--- TARGET SPECIFICATIONS ---\n")
        f.write(f"Expression Evaluated       : {expr_clean}\n")
        f.write(f"Total Base-10 Integer Width: {digit_length} digits\n")
        f.write(f"Math Engine Engine Core    : gmpy2 (C-GMP Assembly, {n_workers}-way parallel)\n\n")
        f.write("--- VERIFICATION METRICS ---\n")
        f.write(f"Final Combined Verdict     : {final_verdict}\n")
        f.write(f"Total Unique Bases Checked : {k_total}\n")
        f.write(f"False-Positive Error Limit : {comb_str}\n\n")
        f.write("--- HARDWARE ENVIRONMENT SIGNATURE ---\n")
        f.write(f"CPU Model                  : {hw['cpu']}\n")
        f.write(f"System RAM Capacity        : {hw['ram']}\n")
        f.write(f"Host Operating System      : {hw['os']}\n")
        f.write(f"Parallel Worker Processes  : {n_workers}\n\n")
        f.write("--- STAGE 1 LOG EXECUTION ---\n")
        f.write(f"Iterations Assigned        : {k1} rounds\n")
        f.write(f"Execution Started          : {start1.strftime('%Y-%m-%d %H:%M:%S.%f')}\n")
        f.write(f"Execution Completed        : {end1.strftime('%Y-%m-%d %H:%M:%S.%f')}\n")
        f.write(f"Stage 1 Result             : {'Passed' if prime1 else 'Failed'}\n")
        f.write(f"Stage 1 Error Bound        : {p1_str}\n\n")
        f.write("--- STAGE 2 LOG EXECUTION ---\n")
        f.write(f"Iterations Assigned        : {k2} rounds\n")
        f.write(f"Execution Started          : {start2.strftime('%Y-%m-%d %H:%M:%S.%f')}\n")
        f.write(f"Execution Completed        : {end2.strftime('%Y-%m-%d %H:%M:%S.%f')}\n")
        f.write(f"Stage 2 Result             : {'Passed' if prime2 else 'Failed'}\n")
        f.write(f"Stage 2 Error Bound        : {p2_str}\n\n")
        f.write("================================================================================\n")
        f.write("                       FULL TRACE AUDIT: BASES ENUMERATED                       \n")
        f.write("================================================================================\n")
        f.write("\n[STAGE 1 WITNESSES]\n")
        for index, b in enumerate(bases1, 1):
            f.write(f"Base 1.{index:03d}: {b}\n")
        f.write("\n[STAGE 2 WITNESSES]\n")
        for index, b in enumerate(bases2, 1):
            f.write(f"Base 2.{index:03d}: {b}\n")

    clear_checkpoint()
    print(f"\nDone! Certificate record generated: '{cert_filename}'")


if __name__ == "__main__":
    main()
