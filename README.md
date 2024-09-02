Took a set of images which is humans are smoking and non smoking images.

Open "https://teachablemachine.withgoogle.com/train" and click on image project -> standerd imnage model.

Now we are going to select the classes. 

As we have 2 classes are given a naming to classes as smoking and non smoking -> click on upload button -> upload the images through local system or from the drive.

once the images are uploaded select advanced options and select Epochs as 1000 and batch sid=ze as 16 learning rate will be 0.001 only. and then click on train model.

once model trained we are going to download it as keras_model.h5 we are going to place this in the project and then you run the server and yoiu test from the local.
