?	?N@a?f@?N@a?f@!?N@a?f@	???x?uM@???x?uM@!???x?uM@"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6?N@a?f@??W?<D@19DܜJn@@A?d?????I?C?3???YG?????Z@*	??|?q	?@2g
0Iterator::Model::MaxIntraOpParallelism::PrefetchF$aߧZ@!Qu;??X@)F$aߧZ@1Qu;??X@:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelism??oD??Z@!?u⬔?X@)?0E?4~??1T??g??:Preprocessing2F
Iterator::Model??׹i?Z@!      Y@)IJzZ?|?13A?b??z?:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
host?Your program is HIGHLY input-bound because 58.9% of the total step time sampled is waiting for input. Therefore, you should first focus on reducing the input time.no*high2t22.0 % of the total step time sampled is spent on 'All Others' time. This could be due to Python execution overhead.9???x?uM@I!@P???6@Q??Ą&2@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	??W?<D@??W?<D@!??W?<D@      ??!       "	9DܜJn@@9DܜJn@@!9DܜJn@@*      ??!       2	?d??????d?????!?d?????:	?C?3????C?3???!?C?3???B      ??!       J	G?????Z@G?????Z@!G?????Z@R      ??!       Z	G?????Z@G?????Z@!G?????Z@b      ??!       JGPUY???x?uM@b q!@P???6@y??Ą&2@?"f
:gradient_tape/model_1/conv2d_3/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter|6U?K^??!|6U?K^??0"f
:gradient_tape/model_1/conv2d_4/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter?????!E?Hb*??0"7
model_1/conv2d_4/Relu_FusedConv2DI?ߙ????!n?T?S??"d
9gradient_tape/model_1/conv2d_5/Conv2D/Conv2DBackpropInputConv2DBackpropInput??$???!O???????0"d
9gradient_tape/model_1/conv2d_4/Conv2D/Conv2DBackpropInputConv2DBackpropInputx??!G??}mL??0"-
IteratorGetNext/_1_Send???5????!?q?????"7
model_1/conv2d_5/Relu_FusedConv2D?"F͢?!9jz2J???"f
:gradient_tape/model_1/conv2d_5/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter!ߚ?a???!+R????0"7
model_1/conv2d_3/Relu_FusedConv2D?`+XoΝ?!0s??????"Z
9gradient_tape/model_1/max_pooling2d_3/MaxPool/MaxPoolGradMaxPoolGrad???QZ???!^?R??w??Q      Y@Yܶm۶m@a?$I?$IW@q???^??y???y???"?

host?Your program is HIGHLY input-bound because 58.9% of the total step time sampled is waiting for input. Therefore, you should first focus on reducing the input time.b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2?
=type.googleapis.com/tensorflow.profiler.GenericRecommendation?
nohigh"t22.0 % of the total step time sampled is spent on 'All Others' time. This could be due to Python execution overhead.*?Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQ2"GPU(: B 