import SimpleITK as sitk


def find_lobes(image: sitk.Image, fissure_seg: sitk.Image, lung_mask: sitk.Image, lobe_scribbles: sitk.Image) -> sitk.Image:
    # post-process fissures
    # make fissure segmentation binary (disregard the 3 different fissures)
    fissure_seg_binary = sitk.BinaryThreshold(fissure_seg, upperThreshold=0.5, insideValue=0, outsideValue=1)

    # create inverted lobe mask by combining fissures and not-lung
    not_lobes = sitk.Or(sitk.Not(lung_mask), fissure_seg_binary)

    # close some gaps
    # not_lobes = sitk.BinaryMorphologicalClosing(not_lobes, kernelRadius=(2, 2, 2), kernelType=sitk.sitkBall)
    not_lobes = sitk.BinaryDilate(not_lobes, kernelRadius=(4, 4, 4), kernelType=sitk.sitkBall)

    # find connected components in lobes mask
    lobes_mask = sitk.Not(not_lobes)
    connected_component_filter = sitk.ConnectedComponentImageFilter()
    lobes_components = connected_component_filter.Execute(lobes_mask)
    print(connected_component_filter.GetObjectCount())

    # find the biggest components (= the 5 lobes)
    # shape_stats = sitk.LabelShapeStatisticsImageFilter()
    # shape_stats.Execute(lobes_components)
    # labels = torch.tensor(shape_stats.GetLabels())
    # object_sizes = torch.tensor([shape_stats.GetPhysicalSize(l.item()) for l in labels])
    # values, indices = torch.topk(object_sizes, k=5)

    # sort objects by size
    relabel_filter = sitk.RelabelComponentImageFilter()
    relabel_filter.SetSortByObjectSize(True)
    lobes_components_sorted = relabel_filter.Execute(lobes_components)
    print(f'The 5 largest objects have sizes {relabel_filter.GetSizeOfObjectsInPhysicalUnits()[:5]}')

    # extract the 5 biggest objects (the 5 lobes)
    change_label_filter = sitk.ChangeLabelImageFilter()
    change_label_filter.SetChangeMap({l: 0 for l in range(6, relabel_filter.GetOriginalNumberOfObjects() + 1)})
    lobes_components_top5 = change_label_filter.Execute(lobes_components_sorted)

    return lobes_components_top5
