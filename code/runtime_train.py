import %%[...]%%
with MultiGpuRuntime("/tmp/train", gpu_devices=[0, 1]) as rt:
  rt.register_model(ConvAutoencoderModel(5e-4))
  rt.register_optimizer(Optimizer("adam", initial_lr=0.001,
                                      step_interval=1000, rate=0.95))	
  rt.register_datasets(MovingMNISTTrainDataset("/tmp/data", as_binary=True,
                                                    input_shape=[10, 64, 64, 1]),
                         MovingMNISTValidDataset("/tmp/data/", as_binary=True,
                                                    input_shape=[10, 64, 64, 1]))
  rt.build(is_autoencoder=True)
  rt.train(batch_size=256, epochs=100,
           valid_batch_size=200, validation_steps=1000)
  # continue training
  def on_valid(rt, global_step):
    inputs, _ = rt.datasets.valid.get_batch(1)
    pred = rt.predict(inputs)
    tl.utils.video.write_multi_gif(os.path.join(rt.train_dir, "{}.gif".format(global_step)),
                                      [inputs[0], pred[0]], fps=5)
  rt.train(batch_size=256, steps=10000, valid_batch_size=200,
            validation_steps=1000, on_validate=on_valid)
