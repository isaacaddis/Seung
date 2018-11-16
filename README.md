# Seung
Neural networks and OpenCV team up together to do my homework.

## Text Detection

The first process Seung undergoes is detecting coherent regions of text using the [https://arxiv.org/abs/1704.03155v2](EAST text detection algorithm).

As with OpenCV4, you can now construct neural networks using the nn module under OpenCV.

## Text Recognition

Seung uses a Connectionist Text Proposal Network (CTPN) to take the regions of text outputted by EAST detection and predict the inner text by localizing lines in a natural image.

## Customization


## Sample Query

```
sudo python3 seung.py -east frozen_east_text_detection.pb -i test_images/Limits1.png
```

