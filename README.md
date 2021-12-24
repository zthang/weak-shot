## Evaluate
Run evaluate_ori.py to evaluate our model, the saved model and the retrieved data is in the saves dir and weight dir, respectively.(suppose you have the novel test image.)

The results are shown in the picture below.

![image-20211224234246114](/Users/zhangtianhang/Library/Application Support/typora-user-images/image-20211224234246114.png)

## Training steps:
Run prepare_graph.py to make the complete graph file.

Run calcu_curvature.py to compute the Ricci-cuvature.

Run gun_pipeline.py to train our RGSimNet model.

Run assign_weight.py to calculate the weight for every novel test image.

Run retrieve_module/write_es.py to construct the retrieve pool.

Run retrieve_module/retrieve_embedding.py to save the retrieved data for every novel test image.

Run train_cls.py to train the main classifier.