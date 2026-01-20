import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt

from skimage.measure import marching_cubes
import plotly.graph_objects as go


def visualize_mask_3d(maskName: str, level: float = 0.5, opacity: float = 0.5):
    """
    Visualize a 3D segmentation mask as a surface mesh using marching cubes.

    Parameters
    ----------
    maskName : str
        Path to the mask file (e.g., .nii or .nii.gz).
    level : float
        Marching cubes threshold. Use 0.5 for binary masks.
    opacity : float
        Opacity of the 3D mesh (0.0 to 1.0).
    """

    msk_itk = sitk.ReadImage(maskName)
    msk_itk = sitk.Cast(msk_itk > 0, sitk.sitkUInt8)

    # Convert to numpy array: shape = (z, y, x)
    msk = sitk.GetArrayFromImage(msk_itk).astype(np.uint8)

    if np.count_nonzero(msk) == 0:
        raise ValueError("Mask is empty (all zeros). Nothing to visualize.")

    spacing_zyx = msk_itk.GetSpacing()[::-1]

    verts, faces, _, _ = marching_cubes(msk, level=level, spacing=spacing_zyx)

    fig = go.Figure(
        data=[
            go.Mesh3d(
                x=verts[:, 2],  # x
                y=verts[:, 1],  # y
                z=verts[:, 0],  # z
                i=faces[:, 0],
                j=faces[:, 1],
                k=faces[:, 2],
                opacity=opacity
            )
        ]
    )

    fig.update_layout(
        title="3D Mask Surface",
        scene=dict(aspectmode="data")
    )

    fig.show()


maskNames = [f"../data/masks/mask_{i}.nii.gz" for i in range(1, 4)]

for maskName in maskNames:
    print(f"Visualizing: {maskName}")
    visualize_mask_3d(maskName)
