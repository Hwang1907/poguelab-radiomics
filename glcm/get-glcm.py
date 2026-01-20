from radiomics import featureextractor
import matplotlib.pyplot as plt
import numpy as np

imageName = "../data/images/image_1.nii.gz"
maskName  = "../data/masks/mask_1.nii.gz"


params = "../params/Params.yaml"

extractor = featureextractor.RadiomicsFeatureExtractor(params)
result = extractor.execute(imageName, maskName)
glcm_only = {k:v for k,v in result.items() if k.startswith("original_glcm")}
print(glcm_only)

val = result["original_glcm_Autocorrelation"]
arr = np.atleast_1d(np.asarray(val)).astype(float)

img = arr.reshape(1, -1)   # 1 x N image

print(arr + ", " + str(arr.size))

plt.imshow(img, aspect="auto")
plt.colorbar(label="Autocorrelation")
plt.yticks([])  # hide y-axis (only one row)
plt.xlabel("Index (direction / distance)")
plt.title("GLCM Autocorrelation heatmap")

plt.show()