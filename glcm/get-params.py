#imports
from radiomics import featureextractor
import pandas as pd
import numpy as np

def extract_volume_glcm(extractor, image_path: str, mask_path: str) -> dict:
    """ 
    Runs PyRadiomics on (image, mask) and returns a FLAT dictionary of key-value pairs such that retains information only on the following: 
    - original_shape* 
    - original_glcm* 
    - diagnostics* 
    
    Any numpy scalar/array values are converted to plain Python types. 
    """
    result = extractor.execute(image_path, mask_path)

    wanted = {
        k: v for k, v in result.items()
        if (
            k.startswith("original_shape") or
            k.startswith("original_glcm") or
            k.startswith("original_firstorder") or
            k.startswith("diagnostics")
        )
    }


    flat = {}
    for k, v in wanted.items():
        if isinstance(v, np.generic):
            flat[k] = v.item()
        elif isinstance(v, np.ndarray):
            flat[k] = v.item() if v.size == 1 else v.flatten().tolist()
        else:
            flat[k] = v

    return flat


def run_batch(params_path: str, n_cases: int, out_csv: str):
    extractor = featureextractor.RadiomicsFeatureExtractor(params_path)
    rows = []

    for i in range(1, n_cases + 1):
        image_path = f"../data/images/image_{i}.nii.gz"
        mask_path  = f"../data/masks/mask_{i}.nii.gz"

        row = extract_volume_glcm(extractor, image_path, mask_path)
        row["Patient #"] = i
        rows.append(row)

    df = pd.DataFrame(rows)

    # Put Patient # first
    cols = ["Patient #"] + [c for c in df.columns if c != "Patient #"]
    df = df[cols]

    # Drop the first 9 feature columns after Patient #
    df = pd.concat([df.iloc[:, :1], df.iloc[:, 10:]], axis=1)

    df.to_csv(out_csv, index=False)
    print(f"Saved {len(df)} patients to {out_csv}")
    return df


if __name__ == "__main__":
    params = "../params/Params.yaml"
    run_batch(params_path=params, n_cases=3, out_csv="results/1-3_healthy.csv")