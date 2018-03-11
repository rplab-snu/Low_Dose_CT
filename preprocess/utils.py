import pydicom
import os
import numpy as np
from skimage.transform import radon
from skimage.transform import iradon, iradon_sart
import matplotlib.pyplot as plt
from glob import glob

def load_3d_dcm(dcm_path, parsetype="CT"):
    """
    Load 3D Slice Image
    https://www.kaggle.com/gzuidhof/full-preprocessing-tutorials
    Parameters
    ----------
    dcm_path    : Dicom sequence directory path
    parsetype   : Type of image and modality
                  To specify the modality of image, use '_' symbol
                  For example, in order to get WT image of PET, image_type="PET_WT"
                  If modality is not specified, it will parse all available files

    Return
    ------
    Dicom list sorted by ImagePositionPatient[2]
    Tag (0020, 0032) Image Position (Patient) DS: ['-158.135803', '-179.035797', '-75.699997']
    """

    def _parse_image(dcm_path, image_type, modality="not_specified"):
        """
        image_type  :    Type of image.
                         CT | PET is available for now
                         If you want to add other image types, modify *types* variable
        modality    :    Modality of image.
                         For example, WT / WC / WS / WM.
                         If "not_specified", it will parse all available files
        * This method will automatically remove `-` from header description in order to recognize 'W-T' label style
        """
        types = {"CT": "CT Image Storage", "PET": "Positron Emission Tomography Image Storage"}
        parsed_list = []
        dcm_list = glob(dcm_path + "/*.dcm")
        if modality == "not_specified":
            for dcm in dcm_list:
                read_data = pydicom.dcmread(dcm)
                if read_data.SOPClassUID == types[image_type]:
                    parsed_list.append(dcm)
        else:
            mod_list_from_data = set()
            for dcm in dcm_list:
                read_data = pydicom.dcmread(dcm)
                if read_data.SOPClassUID == types[image_type] and modality in str(read_data.SeriesDescription).replace("-",""):
                    # Note : remove `-` in order to make 'W-T' label style into 'WT' label style
                    parsed_list.append(dcm)
                mod_list_from_data.add(read_data.SeriesDescription)
            if len(parsed_list) == 0:
                print("Error : DCM list is empty. Check the modality labels below.")
                print("  Modality labels from the data:")
                for mod_label in mod_list_from_data:
                    print("  \t" + mod_label, end="\n")
                print("program terminated")
                exit()
        print("{0} {1} Images has been parsed. (modality = {2})".format(len(parsed_list), image_type, modality))
        return parsed_list

    dcms = [pydicom.dcmread(dcm) for dcm in _parse_image(dcm_path, *parsetype.split("_"))]

    dcms.sort(key = lambda x: float(x.ImagePositionPatient[2]))
    try:
        slice_thickness = np.abs(dcms[0].ImagePositionPatient[2] - dcms[1].ImagePositionPatient[2])
    except:
        slice_thickness = np.abs(dcms[0].SliceLocation - dcms[1].SliceLocation)

    for s in dcms:
        s.SliceThickness = slice_thickness

    return dcms

def dcm_to_npy(dcm, norm=False):
    """
    Parameters
    ----------
    dcm : Single Dicom object(2D), List Dicom object(3D)
    norm : If False :[0 4095]:, True :[0 1]:, default False
    Return
    ------
    2D or 3D CT Image(ndarray float32)
    Examples
    --------
    >>> dcm = pydicom.dcmread("ex.dcm")
    >>> npy = dcm_to_npy(dcm, norm=True)
    >>> npy.min(), npy.max()
    >>> 0.0 1.0
    """
    def _dcm_to_npy(dcm, norm):
        # (0028, 0101) Bits Stored US: 16, return value
        dcm_dtype = dcm.BitsAllocated
        pixel = dcm.pixel_array.astype("float32")

        if dcm.RescaleSlope != 1:
            pixel *= dcm.RescaleSlope
        pixel += dcm.RescaleIntercept + 1024
        # if pixel valus is lower than 0, that pixel is NOT USED PIXEL(circular CT)
        np.clip(pixel, 0, 4095, out=pixel)
        return pixel / 4095 if norm is True else pixel

    if type(dcm) is list:
        return np.stack([_dcm_to_npy(d, norm) for d in dcm])
    else:
        return _dcm_to_npy(dcm, norm)

def get_fbp(ct_img, theta, circle=False, sart=False):
    """
    CT Img -> radon -> sinogram -> iradon -> FBP Img
    Using skimage transform method
    Parameters
    ----------
    theta :  projection angle in degree, dtype=np linspace
    circle : boolean, param of radon, default False
    sart : not implemented use only true, boolean, use iradon_sart, default False
    Returns
    -------
    sinogram, normalized fbp [0 1] ndarray
    Examples
    --------
    >>> sinogram, fbp = get_fbp(ct, theta)
    >>> plt.imshow(sinogram), plt.imshow(fbp)
    """
    if ct_img.min() < 0 or ct_img.max() > 1:
        raise ValueError("CT Img Range : [%d %d]"%(ct_img.min(), ct_img.max()))

    sinogram = radon(ct_img, theta=theta, circle=circle)
    if sart is False:
        fbp = iradon(sinogram, theta=theta, circle=circle)
    else:
        # TODO: Check iradon_sart params
        raise NotImplementedError("not implemented sart")
        fbp = iradon_sart(sinogram, theta)

    if fbp.shape[0] != 512 or fbp.shape[1] != 512:
        raise ValueError("FBP Shape : [%d %d]"%(fbp.shape[0], fbp.shape[1]))

    np.clip(fbp, 0, 1, out=fbp)
    return sinogram, fbp

def plot_3d(image, threshold=-300):
    raise NotImplementedError("not implemented")
    # Position the scan upright,
    # so the head of the patient would be at the top facing the camera
    p = image.transpose(2,1,0)

    verts, faces, normals, values = measure.marching_cubes_lewiner(p, threshold, step_size=3)

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Fancy indexing: `verts[faces]` to generate a collection of triangles
    mesh = Poly3DCollection(verts[faces], alpha=0.70)
    face_color = [1, 1, 0.9]
    mesh.set_facecolor(face_color)
    ax.add_collection3d(mesh)

    ax.set_xlim(0, p.shape[0])
    ax.set_ylim(0, p.shape[1])
    ax.set_zlim(0, p.shape[2])

    plt.show()


class Plot2DSlice:
    """
    Plotting a 2D slice
    move slice by `j, k` keys
    Examples
    --------
    >>> dcms = load_3d_dcm(dcm_path)
    >>> npys = dcm_to_npy(dcms)
    >>> Plot2DSlice(npys)
    """
    def _remove_keymap_conflicts(self, new_keys_set):
        for prop in plt.rcParams:
            if prop.startswith('keymap.'):
                keys = plt.rcParams[prop]
                remove_list = set(keys) & new_keys_set
                for key in remove_list:
                    keys.remove(key)

    def __init__(self, volume):
        self._remove_keymap_conflicts({'j', 'k'})
        self.fig, self.ax = plt.subplots()
        self.ax.volume = volume
        self.ax.index = volume.shape[0] // 2
        self.ax.imshow(volume[self.ax.index], cmap='gray')
        self.fig.canvas.mpl_connect('key_press_event', self._process_key)
        plt.show()

    def _process_key(self, event):
        self.ax = self.fig.axes[0]
        if event.key == 'j':
            self._previous_slice()
        elif event.key == 'k':
            self._next_slice()
        self.fig.canvas.draw()

    def _previous_slice(self):
        volume = self.ax.volume
        self.ax.index = (self.ax.index - 1) % volume.shape[0]  # wrap around using %
        self.ax.images[0].set_array(volume[self.ax.index])

    def _next_slice(self):
        volume = self.ax.volume
        self.ax.index = (self.ax.index + 1) % volume.shape[0]
        self.ax.images[0].set_array(volume[self.ax.index])


# For test
if __name__ == "__main__":
    import random
    import os


    from skimage import measure, morphology
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    def range_test(dcm_path):
        pat, num =  dcm_path.split("/")[-4], dcm_path.split("/")[-1][:-4]
        print("Pat:", pat, "Num:", num)
        dcm =  pydicom.dcmread(dcm_path)
        print("  Origin CT Value Range : ", dcm.pixel_array.min(), dcm.pixel_array.max())
        ct_img_not_norm = dcm_to_npy(dcm, norm=False)
        print("  CT[0 4095] Value Range : ", ct_img_not_norm.min(), ct_img_not_norm.max())
        ct_img = dcm_to_npy(dcm, norm=True)
        print("  CT[0 1] Value Range : ", ct_img.min(), ct_img.max())
        theta = np.linspace(0., 180., max(ct_img.shape), endpoint=False)
        print("  Theta shape : ", theta.shape)
        sinogram, fbp = get_fbp(ct_img, theta)
        print("  Sinogram Range", sinogram.min(), sinogram.max(), sinogram.shape)
        print("  inverse Range : ", fbp.min(), fbp.max(), fbp.shape)

    for folder in glob(r"C:\Users\yjh36\Desktop\TMT LAB\FDG-PET2\*"):
        print(folder + " is projected")
        dcm_path = folder

        # dcms = load_3d_dcm(dcm_path, image_type="CT")
        # dcms = load_3d_dcm(dcm_path, image_type="PET")
        dcms = load_3d_dcm(dcm_path, parsetype="PET_WT")

        print(len(dcms))
        npys = dcm_to_npy(dcms)
        print(npys.shape)
        # plot_3d(npys, 400)
        Plot2DSlice(npys)

    """
    dcm_path = "/media/rplab/2EC4179FC417687B/DW_MAR/01_MA_Image/44295853/44295853_0059.DCM"
    range_test(dcm_path)
    exit()
    """

    """
    # CT Lymph Nodes Data
    patient_list = glob("/media/rplab/2EC4179FC417687B/LowDoseCT/00_Data/CT_Lymph_Nodes/*")
    dcm_list = []
    for patient in patient_list:
        dcm_tmp = glob(patient + "/*/*/*.dcm")
        # random.shuffle(dcm_tmp)
        dcm_list += dcm_tmp[::30]
        random.shuffle(dcm_list)
    print("Len dcm_list : ", len(dcm_list))
    for dcm_path in dcm_list:
        # dcm_path = "/media/rplab/2EC4179FC417687B/LowDoseCT/00_Data/CT_Lymph_Nodes/ABD_LYMPH_002/09-14-2014-ABDLYMPH002-abdominallymphnodes-40168/abdominallymphnodes-68655/000343.dcm"
        range_test(dcm_path)
    """