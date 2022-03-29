import SimpleITK as sitk
from data import LungData


def apply_mask(fissures, mask):
    print('Masking fissures with lung mask.')
    inv_mask = sitk.BinaryNot(mask)  # inverted lung mask
    inv_mask *= 3  # make binary values 0 and 3
    fissures = sitk.Cast(fissures, sitk.sitkInt16) - sitk.Cast(inv_mask, sitk.sitkInt16)  # subtract so that masked out values are <= 0
    fissures = sitk.Cast(sitk.Clamp(fissures, lowerBound=0), sitk.sitkUInt8)  # values < 0 are set to 0
    return fissures


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
