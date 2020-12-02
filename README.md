# GAN-STEM-Conv2MultiSlice
GAN method to help covert lower resolution STEM images generated by convolution methods to higher resolution STEM images generated by Multi-Slice methods.

## Files Usage

### `Calculate_NRMSE.py`

`Calculate_NRMSE.py` helps calculate the normalized root mean square error of two images generated by different methods.

To run the command you need first download the data file from [Google Drive](https://drive.google.com/open?id=1CnIuRKp2C4pYJgAEe4ZPvQWVurJA9Ybi) and name the folder `Data`. Then you can run the program with this command.


```bash
python Calculate_NRMSE.py
```

You need `Python 3` and `scikit-image`,`numpy` and `matplotlib` to run the program .


To change to a different fold of images just comment currently used folder path then uncomment the previously commented folder path.

## GAN

Based on [Keras-GAN](https://github.com/eriklindernoren/Keras-GAN) we have obtained the initial results which shows very promising results.

![GAN initial results of 200 Epoch](/pix2pix/images/stem/199_0.png)