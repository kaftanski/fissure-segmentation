import SimpleITK as sitk

from data import LungData
from utils.image_ops import apply_mask

if __name__ == '__main__':
    ds = LungData(folder="../data")

    for i in range(len(ds)):
        if ds.get_fissures(i) is None:
            continue

        print(ds.get_filename(i))

        mask = ds.get_lung_mask(i)
        fissures = sitk.ReadImage(ds.get_filename(i).replace('_img_', '_fissures_poisson_'), outputPixelType=sitk.sitkInt16)

        fissures = apply_mask(fissures, mask)

        sitk.WriteImage(fissures, ds.get_filename(i).replace('_img_', '_fissures_poisson_'))
