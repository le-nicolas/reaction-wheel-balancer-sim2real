import sys

import final


def run_case(name, args, expected):
    sys.argv = ["final.py", *args]
    cfg = final.build_config(final.parse_args())
    actual = {
        "wheel_only": cfg.wheel_only,
        "wheel_only_forced": cfg.wheel_only_forced,
        "allow_base_motion_requested": cfg.allow_base_motion_requested,
        "real_hardware_base_unlocked": cfg.real_hardware_base_unlocked,
        "allow_base_motion": cfg.allow_base_motion,
    }
    ok = all(actual[k] == v for k, v in expected.items())
    status = "PASS" if ok else "FAIL"
    print(f"{status} {name}")
    for key in expected:
        print(f"  {key}: expected={expected[key]} actual={actual[key]}")
    return ok


def main():
    cases = [
        (
            "default",
            [],
            {
                "wheel_only": False,
                "wheel_only_forced": False,
                "allow_base_motion_requested": True,
                "real_hardware_base_unlocked": True,
                "allow_base_motion": True,
            },
        ),
        (
            "wheel_only",
            ["--wheel-only"],
            {
                "wheel_only": True,
                "wheel_only_forced": False,
                "allow_base_motion_requested": True,
                "real_hardware_base_unlocked": True,
                "allow_base_motion": True,
            },
        ),
        (
            "allow_base_motion",
            ["--allow-base-motion"],
            {
                "wheel_only": False,
                "wheel_only_forced": False,
                "allow_base_motion_requested": True,
                "real_hardware_base_unlocked": True,
                "allow_base_motion": True,
            },
        ),
        (
            "real_hardware_locked",
            ["--real-hardware", "--no-unlock-base"],
            {
                "wheel_only": True,
                "wheel_only_forced": True,
                "allow_base_motion_requested": True,
                "real_hardware_base_unlocked": False,
                "allow_base_motion": False,
            },
        ),
        (
            "real_hardware_unlock_allow",
            ["--real-hardware", "--unlock-base", "--allow-base-motion"],
            {
                "wheel_only": False,
                "wheel_only_forced": False,
                "allow_base_motion_requested": True,
                "real_hardware_base_unlocked": True,
                "allow_base_motion": True,
            },
        ),
    ]

    all_ok = True
    for case in cases:
        all_ok = run_case(*case) and all_ok

    if not all_ok:
        raise SystemExit(1)
    print("All mode smoke tests passed.")


if __name__ == "__main__":
    main()
