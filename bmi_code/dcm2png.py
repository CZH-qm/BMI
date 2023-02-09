import os
import numpy as np
import SimpleITK as sitk
from tqdm import trange
from PIL import Image
import torchio as tio


def get_listdir(path):
    dcm_list = []
    tmp_list = os.listdir(path)
    for file in tmp_list:
        dcm_path = os.path.join(path, file, os.listdir(os.path.join(path, file))[0])
        if os.path.splitext(dcm_path)[1] == '.dcm':
            dcm_list.append(dcm_path)
    return dcm_list


def pad(img):
    img_arr = sitk.GetArrayFromImage(img)
    img_arr = np.expand_dims(img_arr, axis=0)
    pad_transform = tio.transforms.CropOrPad((1, 512, 512), padding_mode=-1024)
    new_arr = pad_transform(img_arr)
    new_arr = np.squeeze(new_arr, 0)
    new_img = sitk.GetImageFromArray(new_arr)
    new_img.SetDirection(img.GetDirection())
    new_img.SetOrigin(img.GetOrigin())
    new_img.SetSpacing(img.GetSpacing())
    return new_img


def resample(ct_path, save_path):
    file = sitk.ReadImage(ct_path)
    img_shape = file.GetSize()
    img_spacing = file.GetSpacing()
    resample = sitk.ResampleImageFilter()
    resample.SetInterpolator(sitk.sitkLinear)  # image重采样使用线性插值
    resample.SetDefaultPixelValue(-1000)
    newspacing = [1, 1, 1]
    resample.SetOutputSpacing(newspacing)
    resample.SetOutputOrigin(file.GetOrigin())
    resample.SetOutputDirection(file.GetDirection())
    new_size = np.asarray(img_shape) * np.asarray(img_spacing) / np.asarray(newspacing)
    new_size = new_size.astype(int).tolist()
    resample.SetSize(new_size)
    new = resample.Execute(file)
    new = pad(new)
    img_arr = sitk.GetArrayFromImage(new)

    MIN_BOUND = -600.0  # TODO:修改窗宽窗位
    MAX_BOUND = 600.0  # TODO:修改窗宽窗位
    img_arr[img_arr > MAX_BOUND] = MAX_BOUND
    img_arr[img_arr < MIN_BOUND] = MIN_BOUND
    img_arr = (img_arr - MIN_BOUND) / (MAX_BOUND - MIN_BOUND) * 255
    temp = img_arr[0, :, :].astype(np.uint8)
    img_pil = Image.fromarray(temp)

    path, _ = os.path.split(ct_path)
    dir = path.split('\\')[-1]
    os.makedirs(os.path.join(save_path, dir))
    img_pil.save(os.path.join(save_path, dir, '1.png'))


if __name__ == '__main__':
    # 原始数据，不能有中文
    main_path = r'F:\my_code\bmi_code\bmi_1000'
    save_path = r'F:\my_code\bmi_code\bmi_1000_split'
    ct_path = get_listdir(main_path)
    ct_path.sort()
    for i in trange(len(ct_path)):
        resample(ct_path[i], save_path)
