with DefaultRuntime("/tmp/train") as rt:
  rt.register_model(ConvAutoencoderModel())
  rt.register_datasets(test_ds=MovingMNISTTestDataset("/tmp/data", as_binary=True,
                                                            input_shape=[10, 64, 64, 1]))
  rt.build(is_autoencoder=True, restore_model_params=True
           restore_checkpoint=tl.core.LATEST_CHECKPOINT)
  rt.test(batch_size=100, epochs=100)
