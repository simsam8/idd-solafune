[Link to implementation example](https://tianjinteda.github.io/Transunet.html)


## Recap Questions

### What new aspect(s) does TransUNet introduce compared to UNet?

- [] Cascaded Upsampler and Skip connections
- [] CNN-Transformer encoder
- [x] CNN-Transformer encoder and Cascaded Upsampler(CUP)
- [] Patch Embeddings


### How does the Hybrid encoder(CNN-Transformer) in TransUNet improve performance?

- [x] Capture better global context while keeping low-level details
- [] It is less computationally expensive
- [] By applying an auto-encoder to learn relevant features
- [] All of the above


## Code references

- [UNet using resnet50](https://github.com/rawmarshmellows/pytorch-unet-resnet-50-encoder/blob/master/u_net_resnet_50_encoder.py)
- [Official TransUNet repo](https://github.com/Beckschen/TransUNet/tree/main)
