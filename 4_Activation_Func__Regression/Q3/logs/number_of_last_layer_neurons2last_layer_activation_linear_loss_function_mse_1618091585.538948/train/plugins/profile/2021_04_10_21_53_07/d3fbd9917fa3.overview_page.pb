?	^,?S?e@^,?S?e@!^,?S?e@	I??c2ON@I??c2ON@!I??c2ON@"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6^,?S?e@?V????@@1?T1s@@A?<?????Ig??j+v??Y?? `Z@*	y?&1???@2g
0Iterator::Model::MaxIntraOpParallelism::Prefetch??@ RZ@!?}_~?X@)??@ RZ@1?}_~?X@:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelism????NSZ@!!?{X??X@)6?:???1??2?A???:Preprocessing2F
Iterator::Model???w?SZ@!      Y@)Q??9?y?1?÷??x?:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
host?Your program is HIGHLY input-bound because 60.6% of the total step time sampled is waiting for input. Therefore, you should first focus on reducing the input time.no*high2t19.5 % of the total step time sampled is spent on 'All Others' time. This could be due to Python execution overhead.9J??c2ON@IG??Oz4@Q%R*L?2@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?V????@@?V????@@!?V????@@      ??!       "	?T1s@@?T1s@@!?T1s@@*      ??!       2	?<??????<?????!?<?????:	g??j+v??g??j+v??!g??j+v??B      ??!       J	?? `Z@?? `Z@!?? `Z@R      ??!       Z	?? `Z@?? `Z@!?? `Z@b      ??!       JGPUYJ??c2ON@b qG??Oz4@y%R*L?2@?"f
:gradient_tape/model_3/conv2d_9/Conv2D/Conv2DBackpropFilterConv2DBackpropFilterq?c?a??!q?c?a??0"g
;gradient_tape/model_3/conv2d_10/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter?#??껳?! ?R???0"8
model_3/conv2d_10/Relu_FusedConv2Dyʜڪɦ?!o#g?????"e
:gradient_tape/model_3/conv2d_11/Conv2D/Conv2DBackpropInputConv2DBackpropInput?}v????!+?Uﳯ??0"e
:gradient_tape/model_3/conv2d_10/Conv2D/Conv2DBackpropInputConv2DBackpropInput?lo᭯??!?????E??0"-
IteratorGetNext/_1_Send?A?a????!???wh???"8
model_3/conv2d_11/Relu_FusedConv2Do;??????!3H?C???"g
;gradient_tape/model_3/conv2d_11/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter0???ϋ??!6?s ???0"7
model_3/conv2d_9/Relu_FusedConv2DE,??:???!???????"=
 RMSprop/RMSprop/update_6/truedivRealDiv0'`?????!??A_???Q      Y@Y#?u?)?@a??g?`W@q@??^??y?˲O????"?

host?Your program is HIGHLY input-bound because 60.6% of the total step time sampled is waiting for input. Therefore, you should first focus on reducing the input time.b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2?
=type.googleapis.com/tensorflow.profiler.GenericRecommendation?
nohigh"t19.5 % of the total step time sampled is spent on 'All Others' time. This could be due to Python execution overhead.*?Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQ2"GPU(: B 