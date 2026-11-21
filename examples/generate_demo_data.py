from pathlib import Path

import numpy as np
import pandas as pd


def main() -> None:
    rng = np.random.default_rng(42)
    rows = 600
    load = rng.normal(60, 12, rows)
    speed = rng.normal(1450, 120, rows)
    vibration = 0.02 * load + rng.normal(0, 0.25, rows)
    temperature = 35 + 0.18 * load + 2.5 * vibration + rng.normal(0, 1.5, rows)
    pressure = 2.2 + 0.004 * speed + rng.normal(0, 0.25, rows)
    fault_score = 0.09 * (temperature - 45) + 0.8 * (vibration - 1.2)
    fault = (fault_score + rng.normal(0, 0.5, rows) > 0.5).astype(int)
    remaining_time = 240 - 1.8 * temperature - 22 * vibration + rng.normal(0, 6, rows)
    frame = pd.DataFrame(
        {
            "load": load,
            "speed": speed,
            "vibration": vibration,
            "temperature": temperature,
            "pressure": pressure,
            "fault": fault,
            "remaining_time": remaining_time,
        }
    )
    output = Path("data/demo_equipment.csv")
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False)
    print(f"Generated {len(frame)} rows at {output}")


if __name__ == "__main__":
    main()
