# **_SPDPvCNN_**

**_`Stock Price and Direction Prediction via Deep Attention-Based Convolutional Neural Networks`_**

**_This is the repository for the "Stock Price and Direction Prediction via Deep Attention-Based Convolutional Neural Networks" CS 401 and CS 402 Senior Project at Ozyegin University. This is where you'll find all of the essential materials, such as code, data, and articles._**

## **_Institution_**

- **_[Ozyegin University](https://www.ozyegin.edu.tr/)_**

## **_Project Members_**

- **_[Onur Alaçam](https://github.com/Onralcm)_**<br/>
- **_[Tuğcan Hoşer](https://github.com/Tugcannn)_**<br/>
- **_[Uygar Kaya](https://github.com/UygarKAYA)_**<br/>
- **_[Tuna Tuncer](https://github.com/kuantuna)_**

## **_Project Supervisor_**

- **_[Assistant Prof. Emre Sefer](http://www.emresefer.com/)_**

## **_Dependencies Used_**

In order to run `convmixer.py`, `vision_transformer.py`, and `mlp_mixer.py` one must have the exact versions of dependencies below. Note that the code is incompatible with versions of tensorflow 2.5.0, 2.7.0, and 2.8.0.

- Keras: 2.6.0
- Tensorflow: 2.6.0
- Tensorflow Addons: 0.16.1

## **_Steps to Reproduce Our Results_**

1. We are using `create_labels_images.py` to create images and labels for the specified range of dates.
2. Later, we are using `convmixer.py` to train a model using the images created on the previous phase.
3. Finally, we are using `financial_evaluation.py` to evaluate our model on the test data financially.
