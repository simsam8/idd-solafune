
SegFormer [@xie2021segformer] is a powerful semantic segmentation model that combines the strengths of hierarchical Transformer-based encoders with lightweight MLP decoders. It provides excellent performance across a wide range of segmentation tasks while maintaining computational efficiency.
For this project, SegFormer was implemented through the segmentation_models.pytorch library, which offers ready-to-use, modular segmentation architectures. The encoder-decoder architecture, training pipeline, and model configuration were adapted for the Solafune dataset.

I chose to use the SegFormer-B5, which is the most powerful variant of the architecture. This version has:
- The deepest encoder (Mix Transformer B5), capable of capturing both fine-grained and global features.
- Proven top performance across multiple segmentation tasks.
- The best trade-off between accuracy and model complexity for advanced segmentation tasks like deforestation analysis.

I implemented SegFormer using the [`segmentation_models.pytorch`](https://github.com/qubvel/segmentation_models.pytorch) library. This library provides modular support for many segmentation models and backbones, including SegFormer, and made it straightforward to integrate the model into our training pipeline.


