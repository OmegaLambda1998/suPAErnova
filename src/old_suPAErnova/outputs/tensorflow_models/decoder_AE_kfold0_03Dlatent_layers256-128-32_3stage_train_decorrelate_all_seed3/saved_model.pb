��
��
8
Const
output"dtype"
valuetensor"
dtypetype

NoOp
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype�
�
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring �
q
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape�"serve*2.2.02unknown8��
|
dense_476/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *!
shared_namedense_476/kernel
u
$dense_476/kernel/Read/ReadVariableOpReadVariableOpdense_476/kernel*
_output_shapes

: *
dtype0
t
dense_476/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namedense_476/bias
m
"dense_476/bias/Read/ReadVariableOpReadVariableOpdense_476/bias*
_output_shapes
: *
dtype0
�
color_law_62/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�*$
shared_namecolor_law_62/kernel
|
'color_law_62/kernel/Read/ReadVariableOpReadVariableOpcolor_law_62/kernel*
_output_shapes
:	�*
dtype0
}
dense_477/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	 �*!
shared_namedense_477/kernel
v
$dense_477/kernel/Read/ReadVariableOpReadVariableOpdense_477/kernel*
_output_shapes
:	 �*
dtype0
u
dense_477/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*
shared_namedense_477/bias
n
"dense_477/bias/Read/ReadVariableOpReadVariableOpdense_477/bias*
_output_shapes	
:�*
dtype0
~
dense_478/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
��*!
shared_namedense_478/kernel
w
$dense_478/kernel/Read/ReadVariableOpReadVariableOpdense_478/kernel* 
_output_shapes
:
��*
dtype0
u
dense_478/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*
shared_namedense_478/bias
n
"dense_478/bias/Read/ReadVariableOpReadVariableOpdense_478/bias*
_output_shapes	
:�*
dtype0
~
dense_479/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
��*!
shared_namedense_479/kernel
w
$dense_479/kernel/Read/ReadVariableOpReadVariableOpdense_479/kernel* 
_output_shapes
:
��*
dtype0
u
dense_479/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*
shared_namedense_479/bias
n
"dense_479/bias/Read/ReadVariableOpReadVariableOpdense_479/bias*
_output_shapes	
:�*
dtype0

NoOpNoOp
�:
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*�:
value�9B�9 B�9
�
layer-0
layer-1
layer-2
layer-3
layer-4
layer-5
layer-6
layer-7
	layer_with_weights-0
	layer-8

layer_with_weights-1

layer-9
layer-10
layer_with_weights-2
layer-11
layer-12
layer_with_weights-3
layer-13
layer-14
layer_with_weights-4
layer-15
layer-16
layer-17
layer-18
layer-19
layer-20
layer-21
	variables
regularization_losses
trainable_variables
	keras_api

signatures
 
R
	variables
regularization_losses
trainable_variables
	keras_api
 
R
 	variables
!regularization_losses
"trainable_variables
#	keras_api
R
$	variables
%regularization_losses
&trainable_variables
'	keras_api
R
(	variables
)regularization_losses
*trainable_variables
+	keras_api
R
,	variables
-regularization_losses
.trainable_variables
/	keras_api
R
0	variables
1regularization_losses
2trainable_variables
3	keras_api
h

4kernel
5bias
6	variables
7regularization_losses
8trainable_variables
9	keras_api
^

:kernel
;	variables
<regularization_losses
=trainable_variables
>	keras_api
R
?	variables
@regularization_losses
Atrainable_variables
B	keras_api
h

Ckernel
Dbias
E	variables
Fregularization_losses
Gtrainable_variables
H	keras_api
R
I	variables
Jregularization_losses
Ktrainable_variables
L	keras_api
h

Mkernel
Nbias
O	variables
Pregularization_losses
Qtrainable_variables
R	keras_api
R
S	variables
Tregularization_losses
Utrainable_variables
V	keras_api
h

Wkernel
Xbias
Y	variables
Zregularization_losses
[trainable_variables
\	keras_api
R
]	variables
^regularization_losses
_trainable_variables
`	keras_api
R
a	variables
bregularization_losses
ctrainable_variables
d	keras_api
 
R
e	variables
fregularization_losses
gtrainable_variables
h	keras_api
R
i	variables
jregularization_losses
ktrainable_variables
l	keras_api
R
m	variables
nregularization_losses
otrainable_variables
p	keras_api
?
40
51
:2
C3
D4
M5
N6
W7
X8
 
8
40
51
C2
D3
M4
N5
W6
X7
�
qmetrics
rlayer_regularization_losses
	variables
regularization_losses
trainable_variables
snon_trainable_variables
tlayer_metrics

ulayers
 
 
 
 
�
vmetrics
wlayer_regularization_losses
	variables
regularization_losses
trainable_variables
xnon_trainable_variables
ylayer_metrics

zlayers
 
 
 
�
{metrics
|layer_regularization_losses
 	variables
!regularization_losses
"trainable_variables
}non_trainable_variables
~layer_metrics

layers
 
 
 
�
�metrics
 �layer_regularization_losses
$	variables
%regularization_losses
&trainable_variables
�non_trainable_variables
�layer_metrics
�layers
 
 
 
�
�metrics
 �layer_regularization_losses
(	variables
)regularization_losses
*trainable_variables
�non_trainable_variables
�layer_metrics
�layers
 
 
 
�
�metrics
 �layer_regularization_losses
,	variables
-regularization_losses
.trainable_variables
�non_trainable_variables
�layer_metrics
�layers
 
 
 
�
�metrics
 �layer_regularization_losses
0	variables
1regularization_losses
2trainable_variables
�non_trainable_variables
�layer_metrics
�layers
\Z
VARIABLE_VALUEdense_476/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEdense_476/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE

40
51
 

40
51
�
�metrics
 �layer_regularization_losses
6	variables
7regularization_losses
8trainable_variables
�non_trainable_variables
�layer_metrics
�layers
_]
VARIABLE_VALUEcolor_law_62/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE

:0
 
 
�
�metrics
 �layer_regularization_losses
;	variables
<regularization_losses
=trainable_variables
�non_trainable_variables
�layer_metrics
�layers
 
 
 
�
�metrics
 �layer_regularization_losses
?	variables
@regularization_losses
Atrainable_variables
�non_trainable_variables
�layer_metrics
�layers
\Z
VARIABLE_VALUEdense_477/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEdense_477/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE

C0
D1
 

C0
D1
�
�metrics
 �layer_regularization_losses
E	variables
Fregularization_losses
Gtrainable_variables
�non_trainable_variables
�layer_metrics
�layers
 
 
 
�
�metrics
 �layer_regularization_losses
I	variables
Jregularization_losses
Ktrainable_variables
�non_trainable_variables
�layer_metrics
�layers
\Z
VARIABLE_VALUEdense_478/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEdense_478/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE

M0
N1
 

M0
N1
�
�metrics
 �layer_regularization_losses
O	variables
Pregularization_losses
Qtrainable_variables
�non_trainable_variables
�layer_metrics
�layers
 
 
 
�
�metrics
 �layer_regularization_losses
S	variables
Tregularization_losses
Utrainable_variables
�non_trainable_variables
�layer_metrics
�layers
\Z
VARIABLE_VALUEdense_479/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEdense_479/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE

W0
X1
 

W0
X1
�
�metrics
 �layer_regularization_losses
Y	variables
Zregularization_losses
[trainable_variables
�non_trainable_variables
�layer_metrics
�layers
 
 
 
�
�metrics
 �layer_regularization_losses
]	variables
^regularization_losses
_trainable_variables
�non_trainable_variables
�layer_metrics
�layers
 
 
 
�
�metrics
 �layer_regularization_losses
a	variables
bregularization_losses
ctrainable_variables
�non_trainable_variables
�layer_metrics
�layers
 
 
 
�
�metrics
 �layer_regularization_losses
e	variables
fregularization_losses
gtrainable_variables
�non_trainable_variables
�layer_metrics
�layers
 
 
 
�
�metrics
 �layer_regularization_losses
i	variables
jregularization_losses
ktrainable_variables
�non_trainable_variables
�layer_metrics
�layers
 
 
 
�
�metrics
 �layer_regularization_losses
m	variables
nregularization_losses
otrainable_variables
�non_trainable_variables
�layer_metrics
�layers
 
 

:0
 
�
0
1
2
3
4
5
6
7
	8

9
10
11
12
13
14
15
16
17
18
19
20
21
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 

:0
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
�
"serving_default_conditional_paramsPlaceholder*+
_output_shapes
:��������� *
dtype0* 
shape:��������� 
�
serving_default_input_240Placeholder*,
_output_shapes
:��������� �*
dtype0*!
shape:��������� �
�
serving_default_latent_paramsPlaceholder*'
_output_shapes
:���������*
dtype0*
shape:���������
�
StatefulPartitionedCallStatefulPartitionedCall"serving_default_conditional_paramsserving_default_input_240serving_default_latent_paramscolor_law_62/kerneldense_476/kerneldense_476/biasdense_477/kerneldense_477/biasdense_478/kerneldense_478/biasdense_479/kerneldense_479/bias*
Tin
2*
Tout
2*,
_output_shapes
:��������� �*+
_read_only_resource_inputs
		
*-
config_proto

GPU

CPU2*0J 8*-
f(R&
$__inference_signature_wrapper_462158
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
�
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename$dense_476/kernel/Read/ReadVariableOp"dense_476/bias/Read/ReadVariableOp'color_law_62/kernel/Read/ReadVariableOp$dense_477/kernel/Read/ReadVariableOp"dense_477/bias/Read/ReadVariableOp$dense_478/kernel/Read/ReadVariableOp"dense_478/bias/Read/ReadVariableOp$dense_479/kernel/Read/ReadVariableOp"dense_479/bias/Read/ReadVariableOpConst*
Tin
2*
Tout
2*
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

GPU

CPU2*0J 8*(
f#R!
__inference__traced_save_462951
�
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedense_476/kerneldense_476/biascolor_law_62/kerneldense_477/kerneldense_477/biasdense_478/kerneldense_478/biasdense_479/kerneldense_479/bias*
Tin
2
*
Tout
2*
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

GPU

CPU2*0J 8*+
f&R$
"__inference__traced_restore_462990��
�
�
E__inference_dense_478_layer_call_and_return_conditional_losses_461817

inputs%
!tensordot_readvariableop_resource#
biasadd_readvariableop_resource
identity��
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource* 
_output_shapes
:
��*
dtype02
Tensordot/ReadVariableOpj
Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
Tensordot/axesq
Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
Tensordot/freeX
Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:2
Tensordot/Shapet
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/GatherV2/axis�
Tensordot/GatherV2GatherV2Tensordot/Shape:output:0Tensordot/free:output:0 Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
Tensordot/GatherV2x
Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/GatherV2_1/axis�
Tensordot/GatherV2_1GatherV2Tensordot/Shape:output:0Tensordot/axes:output:0"Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
Tensordot/GatherV2_1l
Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot/Const�
Tensordot/ProdProdTensordot/GatherV2:output:0Tensordot/Const:output:0*
T0*
_output_shapes
: 2
Tensordot/Prodp
Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot/Const_1�
Tensordot/Prod_1ProdTensordot/GatherV2_1:output:0Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
Tensordot/Prod_1p
Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/concat/axis�
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat�
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
Tensordot/stack�
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*,
_output_shapes
:��������� �2
Tensordot/transpose�
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:������������������2
Tensordot/Reshape�
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
Tensordot/MatMulq
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:�2
Tensordot/Const_2t
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/concat_1/axis�
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat_1�
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*,
_output_shapes
:��������� �2
	Tensordot�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02
BiasAdd/ReadVariableOp�
BiasAddAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:��������� �2	
BiasAddX
ReluReluBiasAdd:z:0*
T0*,
_output_shapes
:��������� �2
Reluk
IdentityIdentityRelu:activations:0*
T0*,
_output_shapes
:��������� �2

Identity"
identityIdentity:output:0*3
_input_shapes"
 :��������� �:::T P
,
_output_shapes
:��������� �
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
�K
�
E__inference_model_119_layer_call_and_return_conditional_losses_462042

inputs
inputs_1
inputs_2
color_law_462010
dense_476_462014
dense_476_462016
dense_477_462020
dense_477_462022
dense_478_462026
dense_478_462028
dense_479_462031
dense_479_462033
identity��!color_law/StatefulPartitionedCall�!dense_476/StatefulPartitionedCall�!dense_477/StatefulPartitionedCall�!dense_478/StatefulPartitionedCall�!dense_479/StatefulPartitionedCall�
 repeat_vector_59/PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*+
_output_shapes
:��������� * 
_read_only_resource_inputs
 *-
config_proto

GPU

CPU2*0J 8*U
fPRN
L__inference_repeat_vector_59_layer_call_and_return_conditional_losses_4615162"
 repeat_vector_59/PartitionedCall�
-tf_op_layer_strided_slice_480/PartitionedCallPartitionedCall)repeat_vector_59/PartitionedCall:output:0*
Tin
2*
Tout
2*+
_output_shapes
:��������� * 
_read_only_resource_inputs
 *-
config_proto

GPU

CPU2*0J 8*b
f]R[
Y__inference_tf_op_layer_strided_slice_480_layer_call_and_return_conditional_losses_4615372/
-tf_op_layer_strided_slice_480/PartitionedCall�
-tf_op_layer_strided_slice_483/PartitionedCallPartitionedCall)repeat_vector_59/PartitionedCall:output:0*
Tin
2*
Tout
2*+
_output_shapes
:��������� * 
_read_only_resource_inputs
 *-
config_proto

GPU

CPU2*0J 8*b
f]R[
Y__inference_tf_op_layer_strided_slice_483_layer_call_and_return_conditional_losses_4615532/
-tf_op_layer_strided_slice_483/PartitionedCall�
%tf_op_layer_AddV2_118/PartitionedCallPartitionedCallinputs_16tf_op_layer_strided_slice_480/PartitionedCall:output:0*
Tin
2*
Tout
2*+
_output_shapes
:��������� * 
_read_only_resource_inputs
 *-
config_proto

GPU

CPU2*0J 8*Z
fURS
Q__inference_tf_op_layer_AddV2_118_layer_call_and_return_conditional_losses_4615672'
%tf_op_layer_AddV2_118/PartitionedCall�
-tf_op_layer_strided_slice_482/PartitionedCallPartitionedCall)repeat_vector_59/PartitionedCall:output:0*
Tin
2*
Tout
2*+
_output_shapes
:��������� * 
_read_only_resource_inputs
 *-
config_proto

GPU

CPU2*0J 8*b
f]R[
Y__inference_tf_op_layer_strided_slice_482_layer_call_and_return_conditional_losses_4615842/
-tf_op_layer_strided_slice_482/PartitionedCall�
concatenate_179/PartitionedCallPartitionedCall6tf_op_layer_strided_slice_483/PartitionedCall:output:0.tf_op_layer_AddV2_118/PartitionedCall:output:0*
Tin
2*
Tout
2*+
_output_shapes
:��������� * 
_read_only_resource_inputs
 *-
config_proto

GPU

CPU2*0J 8*T
fORM
K__inference_concatenate_179_layer_call_and_return_conditional_losses_4615992!
concatenate_179/PartitionedCall�
!color_law/StatefulPartitionedCallStatefulPartitionedCall6tf_op_layer_strided_slice_482/PartitionedCall:output:0color_law_462010*
Tin
2*
Tout
2*,
_output_shapes
:��������� �*#
_read_only_resource_inputs
*-
config_proto

GPU

CPU2*0J 8*N
fIRG
E__inference_color_law_layer_call_and_return_conditional_losses_4616352#
!color_law/StatefulPartitionedCall�
-tf_op_layer_strided_slice_481/PartitionedCallPartitionedCall)repeat_vector_59/PartitionedCall:output:0*
Tin
2*
Tout
2*+
_output_shapes
:��������� * 
_read_only_resource_inputs
 *-
config_proto

GPU

CPU2*0J 8*b
f]R[
Y__inference_tf_op_layer_strided_slice_481_layer_call_and_return_conditional_losses_4616552/
-tf_op_layer_strided_slice_481/PartitionedCall�
!dense_476/StatefulPartitionedCallStatefulPartitionedCall(concatenate_179/PartitionedCall:output:0dense_476_462014dense_476_462016*
Tin
2*
Tout
2*+
_output_shapes
:���������  *$
_read_only_resource_inputs
*-
config_proto

GPU

CPU2*0J 8*N
fIRG
E__inference_dense_476_layer_call_and_return_conditional_losses_4616942#
!dense_476/StatefulPartitionedCall�
%tf_op_layer_AddV2_119/PartitionedCallPartitionedCall*color_law/StatefulPartitionedCall:output:06tf_op_layer_strided_slice_481/PartitionedCall:output:0*
Tin
2*
Tout
2*,
_output_shapes
:��������� �* 
_read_only_resource_inputs
 *-
config_proto

GPU

CPU2*0J 8*Z
fURS
Q__inference_tf_op_layer_AddV2_119_layer_call_and_return_conditional_losses_4617162'
%tf_op_layer_AddV2_119/PartitionedCall�
!dense_477/StatefulPartitionedCallStatefulPartitionedCall*dense_476/StatefulPartitionedCall:output:0dense_477_462020dense_477_462022*
Tin
2*
Tout
2*,
_output_shapes
:��������� �*$
_read_only_resource_inputs
*-
config_proto

GPU

CPU2*0J 8*N
fIRG
E__inference_dense_477_layer_call_and_return_conditional_losses_4617562#
!dense_477/StatefulPartitionedCall�
#tf_op_layer_Mul_354/PartitionedCallPartitionedCall.tf_op_layer_AddV2_119/PartitionedCall:output:0*
Tin
2*
Tout
2*,
_output_shapes
:��������� �* 
_read_only_resource_inputs
 *-
config_proto

GPU

CPU2*0J 8*X
fSRQ
O__inference_tf_op_layer_Mul_354_layer_call_and_return_conditional_losses_4617782%
#tf_op_layer_Mul_354/PartitionedCall�
!dense_478/StatefulPartitionedCallStatefulPartitionedCall*dense_477/StatefulPartitionedCall:output:0dense_478_462026dense_478_462028*
Tin
2*
Tout
2*,
_output_shapes
:��������� �*$
_read_only_resource_inputs
*-
config_proto

GPU

CPU2*0J 8*N
fIRG
E__inference_dense_478_layer_call_and_return_conditional_losses_4618172#
!dense_478/StatefulPartitionedCall�
!dense_479/StatefulPartitionedCallStatefulPartitionedCall*dense_478/StatefulPartitionedCall:output:0dense_479_462031dense_479_462033*
Tin
2*
Tout
2*,
_output_shapes
:��������� �*$
_read_only_resource_inputs
*-
config_proto

GPU

CPU2*0J 8*N
fIRG
E__inference_dense_479_layer_call_and_return_conditional_losses_4618632#
!dense_479/StatefulPartitionedCall�
"tf_op_layer_Pow_59/PartitionedCallPartitionedCall,tf_op_layer_Mul_354/PartitionedCall:output:0*
Tin
2*
Tout
2*,
_output_shapes
:��������� �* 
_read_only_resource_inputs
 *-
config_proto

GPU

CPU2*0J 8*W
fRRP
N__inference_tf_op_layer_Pow_59_layer_call_and_return_conditional_losses_4618852$
"tf_op_layer_Pow_59/PartitionedCall�
#tf_op_layer_Mul_355/PartitionedCallPartitionedCall*dense_479/StatefulPartitionedCall:output:0+tf_op_layer_Pow_59/PartitionedCall:output:0*
Tin
2*
Tout
2*,
_output_shapes
:��������� �* 
_read_only_resource_inputs
 *-
config_proto

GPU

CPU2*0J 8*X
fSRQ
O__inference_tf_op_layer_Mul_355_layer_call_and_return_conditional_losses_4618992%
#tf_op_layer_Mul_355/PartitionedCall�
#tf_op_layer_Relu_55/PartitionedCallPartitionedCall,tf_op_layer_Mul_355/PartitionedCall:output:0*
Tin
2*
Tout
2*,
_output_shapes
:��������� �* 
_read_only_resource_inputs
 *-
config_proto

GPU

CPU2*0J 8*X
fSRQ
O__inference_tf_op_layer_Relu_55_layer_call_and_return_conditional_losses_4619132%
#tf_op_layer_Relu_55/PartitionedCall�
"tf_op_layer_Max_63/PartitionedCallPartitionedCallinputs_2*
Tin
2*
Tout
2*+
_output_shapes
:��������� * 
_read_only_resource_inputs
 *-
config_proto

GPU

CPU2*0J 8*W
fRRP
N__inference_tf_op_layer_Max_63_layer_call_and_return_conditional_losses_4619272$
"tf_op_layer_Max_63/PartitionedCall�
#tf_op_layer_Mul_356/PartitionedCallPartitionedCall,tf_op_layer_Relu_55/PartitionedCall:output:0+tf_op_layer_Max_63/PartitionedCall:output:0*
Tin
2*
Tout
2*,
_output_shapes
:��������� �* 
_read_only_resource_inputs
 *-
config_proto

GPU

CPU2*0J 8*X
fSRQ
O__inference_tf_op_layer_Mul_356_layer_call_and_return_conditional_losses_4619412%
#tf_op_layer_Mul_356/PartitionedCall�
IdentityIdentity,tf_op_layer_Mul_356/PartitionedCall:output:0"^color_law/StatefulPartitionedCall"^dense_476/StatefulPartitionedCall"^dense_477/StatefulPartitionedCall"^dense_478/StatefulPartitionedCall"^dense_479/StatefulPartitionedCall*
T0*,
_output_shapes
:��������� �2

Identity"
identityIdentity:output:0*y
_input_shapesh
f:���������:��������� :��������� �:::::::::2F
!color_law/StatefulPartitionedCall!color_law/StatefulPartitionedCall2F
!dense_476/StatefulPartitionedCall!dense_476/StatefulPartitionedCall2F
!dense_477/StatefulPartitionedCall!dense_477/StatefulPartitionedCall2F
!dense_478/StatefulPartitionedCall!dense_478/StatefulPartitionedCall2F
!dense_479/StatefulPartitionedCall!dense_479/StatefulPartitionedCall:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs:SO
+
_output_shapes
:��������� 
 
_user_specified_nameinputs:TP
,
_output_shapes
:��������� �
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: 
�
u
Y__inference_tf_op_layer_strided_slice_483_layer_call_and_return_conditional_losses_462567

inputs
identity�
strided_slice_483/beginConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_483/begin
strided_slice_483/endConst*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice_483/end�
strided_slice_483/stridesConst*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_483/strides�
strided_slice_483StridedSliceinputs strided_slice_483/begin:output:0strided_slice_483/end:output:0"strided_slice_483/strides:output:0*
Index0*
T0*
_cloned(*+
_output_shapes
:��������� *
ellipsis_mask*
end_mask2
strided_slice_483r
IdentityIdentitystrided_slice_483:output:0*
T0*+
_output_shapes
:��������� 2

Identity"
identityIdentity:output:0**
_input_shapes
:��������� :S O
+
_output_shapes
:��������� 
 
_user_specified_nameinputs
�
}
Q__inference_tf_op_layer_AddV2_118_layer_call_and_return_conditional_losses_462578
inputs_0
inputs_1
identityx
	AddV2_118AddV2inputs_0inputs_1*
T0*
_cloned(*+
_output_shapes
:��������� 2
	AddV2_118e
IdentityIdentityAddV2_118:z:0*
T0*+
_output_shapes
:��������� 2

Identity"
identityIdentity:output:0*A
_input_shapes0
.:��������� :��������� :U Q
+
_output_shapes
:��������� 
"
_user_specified_name
inputs/0:UQ
+
_output_shapes
:��������� 
"
_user_specified_name
inputs/1
�
�
E__inference_dense_476_layer_call_and_return_conditional_losses_462641

inputs%
!tensordot_readvariableop_resource#
biasadd_readvariableop_resource
identity��
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes

: *
dtype02
Tensordot/ReadVariableOpj
Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
Tensordot/axesq
Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
Tensordot/freeX
Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:2
Tensordot/Shapet
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/GatherV2/axis�
Tensordot/GatherV2GatherV2Tensordot/Shape:output:0Tensordot/free:output:0 Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
Tensordot/GatherV2x
Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/GatherV2_1/axis�
Tensordot/GatherV2_1GatherV2Tensordot/Shape:output:0Tensordot/axes:output:0"Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
Tensordot/GatherV2_1l
Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot/Const�
Tensordot/ProdProdTensordot/GatherV2:output:0Tensordot/Const:output:0*
T0*
_output_shapes
: 2
Tensordot/Prodp
Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot/Const_1�
Tensordot/Prod_1ProdTensordot/GatherV2_1:output:0Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
Tensordot/Prod_1p
Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/concat/axis�
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat�
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
Tensordot/stack�
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*+
_output_shapes
:��������� 2
Tensordot/transpose�
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:������������������2
Tensordot/Reshape�
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� 2
Tensordot/MatMulp
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot/Const_2t
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/concat_1/axis�
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat_1�
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*+
_output_shapes
:���������  2
	Tensordot�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp�
BiasAddAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������  2	
BiasAddW
ReluReluBiasAdd:z:0*
T0*+
_output_shapes
:���������  2
Reluj
IdentityIdentityRelu:activations:0*
T0*+
_output_shapes
:���������  2

Identity"
identityIdentity:output:0*2
_input_shapes!
:��������� :::S O
+
_output_shapes
:��������� 
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
�
P
4__inference_tf_op_layer_Mul_354_layer_call_fn_462800

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*,
_output_shapes
:��������� �* 
_read_only_resource_inputs
 *-
config_proto

GPU

CPU2*0J 8*X
fSRQ
O__inference_tf_op_layer_Mul_354_layer_call_and_return_conditional_losses_4617782
PartitionedCallq
IdentityIdentityPartitionedCall:output:0*
T0*,
_output_shapes
:��������� �2

Identity"
identityIdentity:output:0*+
_input_shapes
:��������� �:T P
,
_output_shapes
:��������� �
 
_user_specified_nameinputs
�
b
6__inference_tf_op_layer_AddV2_118_layer_call_fn_462584
inputs_0
inputs_1
identity�
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*+
_output_shapes
:��������� * 
_read_only_resource_inputs
 *-
config_proto

GPU

CPU2*0J 8*Z
fURS
Q__inference_tf_op_layer_AddV2_118_layer_call_and_return_conditional_losses_4615672
PartitionedCallp
IdentityIdentityPartitionedCall:output:0*
T0*+
_output_shapes
:��������� 2

Identity"
identityIdentity:output:0*A
_input_shapes0
.:��������� :��������� :U Q
+
_output_shapes
:��������� 
"
_user_specified_name
inputs/0:UQ
+
_output_shapes
:��������� 
"
_user_specified_name
inputs/1
�
j
N__inference_tf_op_layer_Max_63_layer_call_and_return_conditional_losses_462878

inputs
identity
Max_63/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
���������2
Max_63/reduction_indices�
Max_63Maxinputs!Max_63/reduction_indices:output:0*
T0*
_cloned(*+
_output_shapes
:��������� *
	keep_dims(2
Max_63g
IdentityIdentityMax_63:output:0*
T0*+
_output_shapes
:��������� 2

Identity"
identityIdentity:output:0*+
_input_shapes
:��������� �:T P
,
_output_shapes
:��������� �
 
_user_specified_nameinputs
�K
�
E__inference_model_119_layer_call_and_return_conditional_losses_461951
latent_params
conditional_params
	input_240
color_law_461644
dense_476_461705
dense_476_461707
dense_477_461767
dense_477_461769
dense_478_461828
dense_478_461830
dense_479_461874
dense_479_461876
identity��!color_law/StatefulPartitionedCall�!dense_476/StatefulPartitionedCall�!dense_477/StatefulPartitionedCall�!dense_478/StatefulPartitionedCall�!dense_479/StatefulPartitionedCall�
 repeat_vector_59/PartitionedCallPartitionedCalllatent_params*
Tin
2*
Tout
2*+
_output_shapes
:��������� * 
_read_only_resource_inputs
 *-
config_proto

GPU

CPU2*0J 8*U
fPRN
L__inference_repeat_vector_59_layer_call_and_return_conditional_losses_4615162"
 repeat_vector_59/PartitionedCall�
-tf_op_layer_strided_slice_480/PartitionedCallPartitionedCall)repeat_vector_59/PartitionedCall:output:0*
Tin
2*
Tout
2*+
_output_shapes
:��������� * 
_read_only_resource_inputs
 *-
config_proto

GPU

CPU2*0J 8*b
f]R[
Y__inference_tf_op_layer_strided_slice_480_layer_call_and_return_conditional_losses_4615372/
-tf_op_layer_strided_slice_480/PartitionedCall�
-tf_op_layer_strided_slice_483/PartitionedCallPartitionedCall)repeat_vector_59/PartitionedCall:output:0*
Tin
2*
Tout
2*+
_output_shapes
:��������� * 
_read_only_resource_inputs
 *-
config_proto

GPU

CPU2*0J 8*b
f]R[
Y__inference_tf_op_layer_strided_slice_483_layer_call_and_return_conditional_losses_4615532/
-tf_op_layer_strided_slice_483/PartitionedCall�
%tf_op_layer_AddV2_118/PartitionedCallPartitionedCallconditional_params6tf_op_layer_strided_slice_480/PartitionedCall:output:0*
Tin
2*
Tout
2*+
_output_shapes
:��������� * 
_read_only_resource_inputs
 *-
config_proto

GPU

CPU2*0J 8*Z
fURS
Q__inference_tf_op_layer_AddV2_118_layer_call_and_return_conditional_losses_4615672'
%tf_op_layer_AddV2_118/PartitionedCall�
-tf_op_layer_strided_slice_482/PartitionedCallPartitionedCall)repeat_vector_59/PartitionedCall:output:0*
Tin
2*
Tout
2*+
_output_shapes
:��������� * 
_read_only_resource_inputs
 *-
config_proto

GPU

CPU2*0J 8*b
f]R[
Y__inference_tf_op_layer_strided_slice_482_layer_call_and_return_conditional_losses_4615842/
-tf_op_layer_strided_slice_482/PartitionedCall�
concatenate_179/PartitionedCallPartitionedCall6tf_op_layer_strided_slice_483/PartitionedCall:output:0.tf_op_layer_AddV2_118/PartitionedCall:output:0*
Tin
2*
Tout
2*+
_output_shapes
:��������� * 
_read_only_resource_inputs
 *-
config_proto

GPU

CPU2*0J 8*T
fORM
K__inference_concatenate_179_layer_call_and_return_conditional_losses_4615992!
concatenate_179/PartitionedCall�
!color_law/StatefulPartitionedCallStatefulPartitionedCall6tf_op_layer_strided_slice_482/PartitionedCall:output:0color_law_461644*
Tin
2*
Tout
2*,
_output_shapes
:��������� �*#
_read_only_resource_inputs
*-
config_proto

GPU

CPU2*0J 8*N
fIRG
E__inference_color_law_layer_call_and_return_conditional_losses_4616352#
!color_law/StatefulPartitionedCall�
-tf_op_layer_strided_slice_481/PartitionedCallPartitionedCall)repeat_vector_59/PartitionedCall:output:0*
Tin
2*
Tout
2*+
_output_shapes
:��������� * 
_read_only_resource_inputs
 *-
config_proto

GPU

CPU2*0J 8*b
f]R[
Y__inference_tf_op_layer_strided_slice_481_layer_call_and_return_conditional_losses_4616552/
-tf_op_layer_strided_slice_481/PartitionedCall�
!dense_476/StatefulPartitionedCallStatefulPartitionedCall(concatenate_179/PartitionedCall:output:0dense_476_461705dense_476_461707*
Tin
2*
Tout
2*+
_output_shapes
:���������  *$
_read_only_resource_inputs
*-
config_proto

GPU

CPU2*0J 8*N
fIRG
E__inference_dense_476_layer_call_and_return_conditional_losses_4616942#
!dense_476/StatefulPartitionedCall�
%tf_op_layer_AddV2_119/PartitionedCallPartitionedCall*color_law/StatefulPartitionedCall:output:06tf_op_layer_strided_slice_481/PartitionedCall:output:0*
Tin
2*
Tout
2*,
_output_shapes
:��������� �* 
_read_only_resource_inputs
 *-
config_proto

GPU

CPU2*0J 8*Z
fURS
Q__inference_tf_op_layer_AddV2_119_layer_call_and_return_conditional_losses_4617162'
%tf_op_layer_AddV2_119/PartitionedCall�
!dense_477/StatefulPartitionedCallStatefulPartitionedCall*dense_476/StatefulPartitionedCall:output:0dense_477_461767dense_477_461769*
Tin
2*
Tout
2*,
_output_shapes
:��������� �*$
_read_only_resource_inputs
*-
config_proto

GPU

CPU2*0J 8*N
fIRG
E__inference_dense_477_layer_call_and_return_conditional_losses_4617562#
!dense_477/StatefulPartitionedCall�
#tf_op_layer_Mul_354/PartitionedCallPartitionedCall.tf_op_layer_AddV2_119/PartitionedCall:output:0*
Tin
2*
Tout
2*,
_output_shapes
:��������� �* 
_read_only_resource_inputs
 *-
config_proto

GPU

CPU2*0J 8*X
fSRQ
O__inference_tf_op_layer_Mul_354_layer_call_and_return_conditional_losses_4617782%
#tf_op_layer_Mul_354/PartitionedCall�
!dense_478/StatefulPartitionedCallStatefulPartitionedCall*dense_477/StatefulPartitionedCall:output:0dense_478_461828dense_478_461830*
Tin
2*
Tout
2*,
_output_shapes
:��������� �*$
_read_only_resource_inputs
*-
config_proto

GPU

CPU2*0J 8*N
fIRG
E__inference_dense_478_layer_call_and_return_conditional_losses_4618172#
!dense_478/StatefulPartitionedCall�
!dense_479/StatefulPartitionedCallStatefulPartitionedCall*dense_478/StatefulPartitionedCall:output:0dense_479_461874dense_479_461876*
Tin
2*
Tout
2*,
_output_shapes
:��������� �*$
_read_only_resource_inputs
*-
config_proto

GPU

CPU2*0J 8*N
fIRG
E__inference_dense_479_layer_call_and_return_conditional_losses_4618632#
!dense_479/StatefulPartitionedCall�
"tf_op_layer_Pow_59/PartitionedCallPartitionedCall,tf_op_layer_Mul_354/PartitionedCall:output:0*
Tin
2*
Tout
2*,
_output_shapes
:��������� �* 
_read_only_resource_inputs
 *-
config_proto

GPU

CPU2*0J 8*W
fRRP
N__inference_tf_op_layer_Pow_59_layer_call_and_return_conditional_losses_4618852$
"tf_op_layer_Pow_59/PartitionedCall�
#tf_op_layer_Mul_355/PartitionedCallPartitionedCall*dense_479/StatefulPartitionedCall:output:0+tf_op_layer_Pow_59/PartitionedCall:output:0*
Tin
2*
Tout
2*,
_output_shapes
:��������� �* 
_read_only_resource_inputs
 *-
config_proto

GPU

CPU2*0J 8*X
fSRQ
O__inference_tf_op_layer_Mul_355_layer_call_and_return_conditional_losses_4618992%
#tf_op_layer_Mul_355/PartitionedCall�
#tf_op_layer_Relu_55/PartitionedCallPartitionedCall,tf_op_layer_Mul_355/PartitionedCall:output:0*
Tin
2*
Tout
2*,
_output_shapes
:��������� �* 
_read_only_resource_inputs
 *-
config_proto

GPU

CPU2*0J 8*X
fSRQ
O__inference_tf_op_layer_Relu_55_layer_call_and_return_conditional_losses_4619132%
#tf_op_layer_Relu_55/PartitionedCall�
"tf_op_layer_Max_63/PartitionedCallPartitionedCall	input_240*
Tin
2*
Tout
2*+
_output_shapes
:��������� * 
_read_only_resource_inputs
 *-
config_proto

GPU

CPU2*0J 8*W
fRRP
N__inference_tf_op_layer_Max_63_layer_call_and_return_conditional_losses_4619272$
"tf_op_layer_Max_63/PartitionedCall�
#tf_op_layer_Mul_356/PartitionedCallPartitionedCall,tf_op_layer_Relu_55/PartitionedCall:output:0+tf_op_layer_Max_63/PartitionedCall:output:0*
Tin
2*
Tout
2*,
_output_shapes
:��������� �* 
_read_only_resource_inputs
 *-
config_proto

GPU

CPU2*0J 8*X
fSRQ
O__inference_tf_op_layer_Mul_356_layer_call_and_return_conditional_losses_4619412%
#tf_op_layer_Mul_356/PartitionedCall�
IdentityIdentity,tf_op_layer_Mul_356/PartitionedCall:output:0"^color_law/StatefulPartitionedCall"^dense_476/StatefulPartitionedCall"^dense_477/StatefulPartitionedCall"^dense_478/StatefulPartitionedCall"^dense_479/StatefulPartitionedCall*
T0*,
_output_shapes
:��������� �2

Identity"
identityIdentity:output:0*y
_input_shapesh
f:���������:��������� :��������� �:::::::::2F
!color_law/StatefulPartitionedCall!color_law/StatefulPartitionedCall2F
!dense_476/StatefulPartitionedCall!dense_476/StatefulPartitionedCall2F
!dense_477/StatefulPartitionedCall!dense_477/StatefulPartitionedCall2F
!dense_478/StatefulPartitionedCall!dense_478/StatefulPartitionedCall2F
!dense_479/StatefulPartitionedCall!dense_479/StatefulPartitionedCall:V R
'
_output_shapes
:���������
'
_user_specified_namelatent_params:_[
+
_output_shapes
:��������� 
,
_user_specified_nameconditional_params:WS
,
_output_shapes
:��������� �
#
_user_specified_name	input_240:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: 
�
�
*__inference_model_119_layer_call_fn_462546
inputs_0
inputs_1
inputs_2
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputs_0inputs_1inputs_2unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7*
Tin
2*
Tout
2*,
_output_shapes
:��������� �*+
_read_only_resource_inputs
		
*-
config_proto

GPU

CPU2*0J 8*N
fIRG
E__inference_model_119_layer_call_and_return_conditional_losses_4621102
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*,
_output_shapes
:��������� �2

Identity"
identityIdentity:output:0*y
_input_shapesh
f:���������:��������� :��������� �:::::::::22
StatefulPartitionedCallStatefulPartitionedCall:Q M
'
_output_shapes
:���������
"
_user_specified_name
inputs/0:UQ
+
_output_shapes
:��������� 
"
_user_specified_name
inputs/1:VR
,
_output_shapes
:��������� �
"
_user_specified_name
inputs/2:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: 
�
y
O__inference_tf_op_layer_Mul_355_layer_call_and_return_conditional_losses_461899

inputs
inputs_1
identityq
Mul_355Mulinputsinputs_1*
T0*
_cloned(*,
_output_shapes
:��������� �2	
Mul_355d
IdentityIdentityMul_355:z:0*
T0*,
_output_shapes
:��������� �2

Identity"
identityIdentity:output:0*C
_input_shapes2
0:��������� �:��������� �:T P
,
_output_shapes
:��������� �
 
_user_specified_nameinputs:TP
,
_output_shapes
:��������� �
 
_user_specified_nameinputs
�

*__inference_dense_477_layer_call_fn_462737

inputs
unknown
	unknown_0
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*,
_output_shapes
:��������� �*$
_read_only_resource_inputs
*-
config_proto

GPU

CPU2*0J 8*N
fIRG
E__inference_dense_477_layer_call_and_return_conditional_losses_4617562
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*,
_output_shapes
:��������� �2

Identity"
identityIdentity:output:0*2
_input_shapes!
:���������  ::22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:���������  
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
��
�
E__inference_model_119_layer_call_and_return_conditional_losses_462327
inputs_0
inputs_1
inputs_2/
+color_law_tensordot_readvariableop_resource/
+dense_476_tensordot_readvariableop_resource-
)dense_476_biasadd_readvariableop_resource/
+dense_477_tensordot_readvariableop_resource-
)dense_477_biasadd_readvariableop_resource/
+dense_478_tensordot_readvariableop_resource-
)dense_478_biasadd_readvariableop_resource/
+dense_479_tensordot_readvariableop_resource-
)dense_479_biasadd_readvariableop_resource
identity��
repeat_vector_59/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2!
repeat_vector_59/ExpandDims/dim�
repeat_vector_59/ExpandDims
ExpandDimsinputs_0(repeat_vector_59/ExpandDims/dim:output:0*
T0*+
_output_shapes
:���������2
repeat_vector_59/ExpandDims�
repeat_vector_59/stackConst*
_output_shapes
:*
dtype0*!
valueB"          2
repeat_vector_59/stack�
repeat_vector_59/TileTile$repeat_vector_59/ExpandDims:output:0repeat_vector_59/stack:output:0*
T0*+
_output_shapes
:��������� 2
repeat_vector_59/Tile�
5tf_op_layer_strided_slice_480/strided_slice_480/beginConst*
_output_shapes
:*
dtype0*
valueB"        27
5tf_op_layer_strided_slice_480/strided_slice_480/begin�
3tf_op_layer_strided_slice_480/strided_slice_480/endConst*
_output_shapes
:*
dtype0*
valueB"       25
3tf_op_layer_strided_slice_480/strided_slice_480/end�
7tf_op_layer_strided_slice_480/strided_slice_480/stridesConst*
_output_shapes
:*
dtype0*
valueB"      29
7tf_op_layer_strided_slice_480/strided_slice_480/strides�
/tf_op_layer_strided_slice_480/strided_slice_480StridedSlicerepeat_vector_59/Tile:output:0>tf_op_layer_strided_slice_480/strided_slice_480/begin:output:0<tf_op_layer_strided_slice_480/strided_slice_480/end:output:0@tf_op_layer_strided_slice_480/strided_slice_480/strides:output:0*
Index0*
T0*
_cloned(*+
_output_shapes
:��������� *
ellipsis_mask21
/tf_op_layer_strided_slice_480/strided_slice_480�
5tf_op_layer_strided_slice_483/strided_slice_483/beginConst*
_output_shapes
:*
dtype0*
valueB"       27
5tf_op_layer_strided_slice_483/strided_slice_483/begin�
3tf_op_layer_strided_slice_483/strided_slice_483/endConst*
_output_shapes
:*
dtype0*
valueB"        25
3tf_op_layer_strided_slice_483/strided_slice_483/end�
7tf_op_layer_strided_slice_483/strided_slice_483/stridesConst*
_output_shapes
:*
dtype0*
valueB"      29
7tf_op_layer_strided_slice_483/strided_slice_483/strides�
/tf_op_layer_strided_slice_483/strided_slice_483StridedSlicerepeat_vector_59/Tile:output:0>tf_op_layer_strided_slice_483/strided_slice_483/begin:output:0<tf_op_layer_strided_slice_483/strided_slice_483/end:output:0@tf_op_layer_strided_slice_483/strided_slice_483/strides:output:0*
Index0*
T0*
_cloned(*+
_output_shapes
:��������� *
ellipsis_mask*
end_mask21
/tf_op_layer_strided_slice_483/strided_slice_483�
tf_op_layer_AddV2_118/AddV2_118AddV2inputs_18tf_op_layer_strided_slice_480/strided_slice_480:output:0*
T0*
_cloned(*+
_output_shapes
:��������� 2!
tf_op_layer_AddV2_118/AddV2_118�
5tf_op_layer_strided_slice_482/strided_slice_482/beginConst*
_output_shapes
:*
dtype0*
valueB"       27
5tf_op_layer_strided_slice_482/strided_slice_482/begin�
3tf_op_layer_strided_slice_482/strided_slice_482/endConst*
_output_shapes
:*
dtype0*
valueB"       25
3tf_op_layer_strided_slice_482/strided_slice_482/end�
7tf_op_layer_strided_slice_482/strided_slice_482/stridesConst*
_output_shapes
:*
dtype0*
valueB"      29
7tf_op_layer_strided_slice_482/strided_slice_482/strides�
/tf_op_layer_strided_slice_482/strided_slice_482StridedSlicerepeat_vector_59/Tile:output:0>tf_op_layer_strided_slice_482/strided_slice_482/begin:output:0<tf_op_layer_strided_slice_482/strided_slice_482/end:output:0@tf_op_layer_strided_slice_482/strided_slice_482/strides:output:0*
Index0*
T0*
_cloned(*+
_output_shapes
:��������� *
ellipsis_mask21
/tf_op_layer_strided_slice_482/strided_slice_482|
concatenate_179/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concatenate_179/concat/axis�
concatenate_179/concatConcatV28tf_op_layer_strided_slice_483/strided_slice_483:output:0#tf_op_layer_AddV2_118/AddV2_118:z:0$concatenate_179/concat/axis:output:0*
N*
T0*+
_output_shapes
:��������� 2
concatenate_179/concat�
"color_law/Tensordot/ReadVariableOpReadVariableOp+color_law_tensordot_readvariableop_resource*
_output_shapes
:	�*
dtype02$
"color_law/Tensordot/ReadVariableOp~
color_law/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
color_law/Tensordot/axes�
color_law/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
color_law/Tensordot/free�
color_law/Tensordot/ShapeShape8tf_op_layer_strided_slice_482/strided_slice_482:output:0*
T0*
_output_shapes
:2
color_law/Tensordot/Shape�
!color_law/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2#
!color_law/Tensordot/GatherV2/axis�
color_law/Tensordot/GatherV2GatherV2"color_law/Tensordot/Shape:output:0!color_law/Tensordot/free:output:0*color_law/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
color_law/Tensordot/GatherV2�
#color_law/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2%
#color_law/Tensordot/GatherV2_1/axis�
color_law/Tensordot/GatherV2_1GatherV2"color_law/Tensordot/Shape:output:0!color_law/Tensordot/axes:output:0,color_law/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2 
color_law/Tensordot/GatherV2_1�
color_law/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
color_law/Tensordot/Const�
color_law/Tensordot/ProdProd%color_law/Tensordot/GatherV2:output:0"color_law/Tensordot/Const:output:0*
T0*
_output_shapes
: 2
color_law/Tensordot/Prod�
color_law/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
color_law/Tensordot/Const_1�
color_law/Tensordot/Prod_1Prod'color_law/Tensordot/GatherV2_1:output:0$color_law/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
color_law/Tensordot/Prod_1�
color_law/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2!
color_law/Tensordot/concat/axis�
color_law/Tensordot/concatConcatV2!color_law/Tensordot/free:output:0!color_law/Tensordot/axes:output:0(color_law/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
color_law/Tensordot/concat�
color_law/Tensordot/stackPack!color_law/Tensordot/Prod:output:0#color_law/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
color_law/Tensordot/stack�
color_law/Tensordot/transpose	Transpose8tf_op_layer_strided_slice_482/strided_slice_482:output:0#color_law/Tensordot/concat:output:0*
T0*+
_output_shapes
:��������� 2
color_law/Tensordot/transpose�
color_law/Tensordot/ReshapeReshape!color_law/Tensordot/transpose:y:0"color_law/Tensordot/stack:output:0*
T0*0
_output_shapes
:������������������2
color_law/Tensordot/Reshape�
color_law/Tensordot/MatMulMatMul$color_law/Tensordot/Reshape:output:0*color_law/Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
color_law/Tensordot/MatMul�
color_law/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:�2
color_law/Tensordot/Const_2�
!color_law/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2#
!color_law/Tensordot/concat_1/axis�
color_law/Tensordot/concat_1ConcatV2%color_law/Tensordot/GatherV2:output:0$color_law/Tensordot/Const_2:output:0*color_law/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
color_law/Tensordot/concat_1�
color_law/TensordotReshape$color_law/Tensordot/MatMul:product:0%color_law/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:��������� �2
color_law/Tensordot�
5tf_op_layer_strided_slice_481/strided_slice_481/beginConst*
_output_shapes
:*
dtype0*
valueB"       27
5tf_op_layer_strided_slice_481/strided_slice_481/begin�
3tf_op_layer_strided_slice_481/strided_slice_481/endConst*
_output_shapes
:*
dtype0*
valueB"       25
3tf_op_layer_strided_slice_481/strided_slice_481/end�
7tf_op_layer_strided_slice_481/strided_slice_481/stridesConst*
_output_shapes
:*
dtype0*
valueB"      29
7tf_op_layer_strided_slice_481/strided_slice_481/strides�
/tf_op_layer_strided_slice_481/strided_slice_481StridedSlicerepeat_vector_59/Tile:output:0>tf_op_layer_strided_slice_481/strided_slice_481/begin:output:0<tf_op_layer_strided_slice_481/strided_slice_481/end:output:0@tf_op_layer_strided_slice_481/strided_slice_481/strides:output:0*
Index0*
T0*
_cloned(*+
_output_shapes
:��������� *
ellipsis_mask21
/tf_op_layer_strided_slice_481/strided_slice_481�
"dense_476/Tensordot/ReadVariableOpReadVariableOp+dense_476_tensordot_readvariableop_resource*
_output_shapes

: *
dtype02$
"dense_476/Tensordot/ReadVariableOp~
dense_476/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
dense_476/Tensordot/axes�
dense_476/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
dense_476/Tensordot/free�
dense_476/Tensordot/ShapeShapeconcatenate_179/concat:output:0*
T0*
_output_shapes
:2
dense_476/Tensordot/Shape�
!dense_476/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2#
!dense_476/Tensordot/GatherV2/axis�
dense_476/Tensordot/GatherV2GatherV2"dense_476/Tensordot/Shape:output:0!dense_476/Tensordot/free:output:0*dense_476/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_476/Tensordot/GatherV2�
#dense_476/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2%
#dense_476/Tensordot/GatherV2_1/axis�
dense_476/Tensordot/GatherV2_1GatherV2"dense_476/Tensordot/Shape:output:0!dense_476/Tensordot/axes:output:0,dense_476/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2 
dense_476/Tensordot/GatherV2_1�
dense_476/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
dense_476/Tensordot/Const�
dense_476/Tensordot/ProdProd%dense_476/Tensordot/GatherV2:output:0"dense_476/Tensordot/Const:output:0*
T0*
_output_shapes
: 2
dense_476/Tensordot/Prod�
dense_476/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
dense_476/Tensordot/Const_1�
dense_476/Tensordot/Prod_1Prod'dense_476/Tensordot/GatherV2_1:output:0$dense_476/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
dense_476/Tensordot/Prod_1�
dense_476/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2!
dense_476/Tensordot/concat/axis�
dense_476/Tensordot/concatConcatV2!dense_476/Tensordot/free:output:0!dense_476/Tensordot/axes:output:0(dense_476/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
dense_476/Tensordot/concat�
dense_476/Tensordot/stackPack!dense_476/Tensordot/Prod:output:0#dense_476/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
dense_476/Tensordot/stack�
dense_476/Tensordot/transpose	Transposeconcatenate_179/concat:output:0#dense_476/Tensordot/concat:output:0*
T0*+
_output_shapes
:��������� 2
dense_476/Tensordot/transpose�
dense_476/Tensordot/ReshapeReshape!dense_476/Tensordot/transpose:y:0"dense_476/Tensordot/stack:output:0*
T0*0
_output_shapes
:������������������2
dense_476/Tensordot/Reshape�
dense_476/Tensordot/MatMulMatMul$dense_476/Tensordot/Reshape:output:0*dense_476/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� 2
dense_476/Tensordot/MatMul�
dense_476/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: 2
dense_476/Tensordot/Const_2�
!dense_476/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2#
!dense_476/Tensordot/concat_1/axis�
dense_476/Tensordot/concat_1ConcatV2%dense_476/Tensordot/GatherV2:output:0$dense_476/Tensordot/Const_2:output:0*dense_476/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
dense_476/Tensordot/concat_1�
dense_476/TensordotReshape$dense_476/Tensordot/MatMul:product:0%dense_476/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:���������  2
dense_476/Tensordot�
 dense_476/BiasAdd/ReadVariableOpReadVariableOp)dense_476_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02"
 dense_476/BiasAdd/ReadVariableOp�
dense_476/BiasAddAdddense_476/Tensordot:output:0(dense_476/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������  2
dense_476/BiasAddu
dense_476/ReluReludense_476/BiasAdd:z:0*
T0*+
_output_shapes
:���������  2
dense_476/Relu�
tf_op_layer_AddV2_119/AddV2_119AddV2color_law/Tensordot:output:08tf_op_layer_strided_slice_481/strided_slice_481:output:0*
T0*
_cloned(*,
_output_shapes
:��������� �2!
tf_op_layer_AddV2_119/AddV2_119�
"dense_477/Tensordot/ReadVariableOpReadVariableOp+dense_477_tensordot_readvariableop_resource*
_output_shapes
:	 �*
dtype02$
"dense_477/Tensordot/ReadVariableOp~
dense_477/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
dense_477/Tensordot/axes�
dense_477/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
dense_477/Tensordot/free�
dense_477/Tensordot/ShapeShapedense_476/Relu:activations:0*
T0*
_output_shapes
:2
dense_477/Tensordot/Shape�
!dense_477/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2#
!dense_477/Tensordot/GatherV2/axis�
dense_477/Tensordot/GatherV2GatherV2"dense_477/Tensordot/Shape:output:0!dense_477/Tensordot/free:output:0*dense_477/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_477/Tensordot/GatherV2�
#dense_477/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2%
#dense_477/Tensordot/GatherV2_1/axis�
dense_477/Tensordot/GatherV2_1GatherV2"dense_477/Tensordot/Shape:output:0!dense_477/Tensordot/axes:output:0,dense_477/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2 
dense_477/Tensordot/GatherV2_1�
dense_477/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
dense_477/Tensordot/Const�
dense_477/Tensordot/ProdProd%dense_477/Tensordot/GatherV2:output:0"dense_477/Tensordot/Const:output:0*
T0*
_output_shapes
: 2
dense_477/Tensordot/Prod�
dense_477/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
dense_477/Tensordot/Const_1�
dense_477/Tensordot/Prod_1Prod'dense_477/Tensordot/GatherV2_1:output:0$dense_477/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
dense_477/Tensordot/Prod_1�
dense_477/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2!
dense_477/Tensordot/concat/axis�
dense_477/Tensordot/concatConcatV2!dense_477/Tensordot/free:output:0!dense_477/Tensordot/axes:output:0(dense_477/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
dense_477/Tensordot/concat�
dense_477/Tensordot/stackPack!dense_477/Tensordot/Prod:output:0#dense_477/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
dense_477/Tensordot/stack�
dense_477/Tensordot/transpose	Transposedense_476/Relu:activations:0#dense_477/Tensordot/concat:output:0*
T0*+
_output_shapes
:���������  2
dense_477/Tensordot/transpose�
dense_477/Tensordot/ReshapeReshape!dense_477/Tensordot/transpose:y:0"dense_477/Tensordot/stack:output:0*
T0*0
_output_shapes
:������������������2
dense_477/Tensordot/Reshape�
dense_477/Tensordot/MatMulMatMul$dense_477/Tensordot/Reshape:output:0*dense_477/Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
dense_477/Tensordot/MatMul�
dense_477/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:�2
dense_477/Tensordot/Const_2�
!dense_477/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2#
!dense_477/Tensordot/concat_1/axis�
dense_477/Tensordot/concat_1ConcatV2%dense_477/Tensordot/GatherV2:output:0$dense_477/Tensordot/Const_2:output:0*dense_477/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
dense_477/Tensordot/concat_1�
dense_477/TensordotReshape$dense_477/Tensordot/MatMul:product:0%dense_477/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:��������� �2
dense_477/Tensordot�
 dense_477/BiasAdd/ReadVariableOpReadVariableOp)dense_477_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02"
 dense_477/BiasAdd/ReadVariableOp�
dense_477/BiasAddAdddense_477/Tensordot:output:0(dense_477/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:��������� �2
dense_477/BiasAddv
dense_477/ReluReludense_477/BiasAdd:z:0*
T0*,
_output_shapes
:��������� �2
dense_477/Relu�
tf_op_layer_Mul_354/Mul_354/xConst*
_output_shapes
: *
dtype0*
valueB
 *��̾2
tf_op_layer_Mul_354/Mul_354/x�
tf_op_layer_Mul_354/Mul_354Mul&tf_op_layer_Mul_354/Mul_354/x:output:0#tf_op_layer_AddV2_119/AddV2_119:z:0*
T0*
_cloned(*,
_output_shapes
:��������� �2
tf_op_layer_Mul_354/Mul_354�
"dense_478/Tensordot/ReadVariableOpReadVariableOp+dense_478_tensordot_readvariableop_resource* 
_output_shapes
:
��*
dtype02$
"dense_478/Tensordot/ReadVariableOp~
dense_478/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
dense_478/Tensordot/axes�
dense_478/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
dense_478/Tensordot/free�
dense_478/Tensordot/ShapeShapedense_477/Relu:activations:0*
T0*
_output_shapes
:2
dense_478/Tensordot/Shape�
!dense_478/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2#
!dense_478/Tensordot/GatherV2/axis�
dense_478/Tensordot/GatherV2GatherV2"dense_478/Tensordot/Shape:output:0!dense_478/Tensordot/free:output:0*dense_478/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_478/Tensordot/GatherV2�
#dense_478/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2%
#dense_478/Tensordot/GatherV2_1/axis�
dense_478/Tensordot/GatherV2_1GatherV2"dense_478/Tensordot/Shape:output:0!dense_478/Tensordot/axes:output:0,dense_478/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2 
dense_478/Tensordot/GatherV2_1�
dense_478/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
dense_478/Tensordot/Const�
dense_478/Tensordot/ProdProd%dense_478/Tensordot/GatherV2:output:0"dense_478/Tensordot/Const:output:0*
T0*
_output_shapes
: 2
dense_478/Tensordot/Prod�
dense_478/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
dense_478/Tensordot/Const_1�
dense_478/Tensordot/Prod_1Prod'dense_478/Tensordot/GatherV2_1:output:0$dense_478/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
dense_478/Tensordot/Prod_1�
dense_478/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2!
dense_478/Tensordot/concat/axis�
dense_478/Tensordot/concatConcatV2!dense_478/Tensordot/free:output:0!dense_478/Tensordot/axes:output:0(dense_478/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
dense_478/Tensordot/concat�
dense_478/Tensordot/stackPack!dense_478/Tensordot/Prod:output:0#dense_478/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
dense_478/Tensordot/stack�
dense_478/Tensordot/transpose	Transposedense_477/Relu:activations:0#dense_478/Tensordot/concat:output:0*
T0*,
_output_shapes
:��������� �2
dense_478/Tensordot/transpose�
dense_478/Tensordot/ReshapeReshape!dense_478/Tensordot/transpose:y:0"dense_478/Tensordot/stack:output:0*
T0*0
_output_shapes
:������������������2
dense_478/Tensordot/Reshape�
dense_478/Tensordot/MatMulMatMul$dense_478/Tensordot/Reshape:output:0*dense_478/Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
dense_478/Tensordot/MatMul�
dense_478/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:�2
dense_478/Tensordot/Const_2�
!dense_478/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2#
!dense_478/Tensordot/concat_1/axis�
dense_478/Tensordot/concat_1ConcatV2%dense_478/Tensordot/GatherV2:output:0$dense_478/Tensordot/Const_2:output:0*dense_478/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
dense_478/Tensordot/concat_1�
dense_478/TensordotReshape$dense_478/Tensordot/MatMul:product:0%dense_478/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:��������� �2
dense_478/Tensordot�
 dense_478/BiasAdd/ReadVariableOpReadVariableOp)dense_478_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02"
 dense_478/BiasAdd/ReadVariableOp�
dense_478/BiasAddAdddense_478/Tensordot:output:0(dense_478/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:��������� �2
dense_478/BiasAddv
dense_478/ReluReludense_478/BiasAdd:z:0*
T0*,
_output_shapes
:��������� �2
dense_478/Relu�
"dense_479/Tensordot/ReadVariableOpReadVariableOp+dense_479_tensordot_readvariableop_resource* 
_output_shapes
:
��*
dtype02$
"dense_479/Tensordot/ReadVariableOp~
dense_479/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
dense_479/Tensordot/axes�
dense_479/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
dense_479/Tensordot/free�
dense_479/Tensordot/ShapeShapedense_478/Relu:activations:0*
T0*
_output_shapes
:2
dense_479/Tensordot/Shape�
!dense_479/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2#
!dense_479/Tensordot/GatherV2/axis�
dense_479/Tensordot/GatherV2GatherV2"dense_479/Tensordot/Shape:output:0!dense_479/Tensordot/free:output:0*dense_479/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_479/Tensordot/GatherV2�
#dense_479/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2%
#dense_479/Tensordot/GatherV2_1/axis�
dense_479/Tensordot/GatherV2_1GatherV2"dense_479/Tensordot/Shape:output:0!dense_479/Tensordot/axes:output:0,dense_479/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2 
dense_479/Tensordot/GatherV2_1�
dense_479/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
dense_479/Tensordot/Const�
dense_479/Tensordot/ProdProd%dense_479/Tensordot/GatherV2:output:0"dense_479/Tensordot/Const:output:0*
T0*
_output_shapes
: 2
dense_479/Tensordot/Prod�
dense_479/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
dense_479/Tensordot/Const_1�
dense_479/Tensordot/Prod_1Prod'dense_479/Tensordot/GatherV2_1:output:0$dense_479/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
dense_479/Tensordot/Prod_1�
dense_479/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2!
dense_479/Tensordot/concat/axis�
dense_479/Tensordot/concatConcatV2!dense_479/Tensordot/free:output:0!dense_479/Tensordot/axes:output:0(dense_479/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
dense_479/Tensordot/concat�
dense_479/Tensordot/stackPack!dense_479/Tensordot/Prod:output:0#dense_479/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
dense_479/Tensordot/stack�
dense_479/Tensordot/transpose	Transposedense_478/Relu:activations:0#dense_479/Tensordot/concat:output:0*
T0*,
_output_shapes
:��������� �2
dense_479/Tensordot/transpose�
dense_479/Tensordot/ReshapeReshape!dense_479/Tensordot/transpose:y:0"dense_479/Tensordot/stack:output:0*
T0*0
_output_shapes
:������������������2
dense_479/Tensordot/Reshape�
dense_479/Tensordot/MatMulMatMul$dense_479/Tensordot/Reshape:output:0*dense_479/Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
dense_479/Tensordot/MatMul�
dense_479/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:�2
dense_479/Tensordot/Const_2�
!dense_479/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2#
!dense_479/Tensordot/concat_1/axis�
dense_479/Tensordot/concat_1ConcatV2%dense_479/Tensordot/GatherV2:output:0$dense_479/Tensordot/Const_2:output:0*dense_479/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
dense_479/Tensordot/concat_1�
dense_479/TensordotReshape$dense_479/Tensordot/MatMul:product:0%dense_479/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:��������� �2
dense_479/Tensordot�
 dense_479/BiasAdd/ReadVariableOpReadVariableOp)dense_479_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02"
 dense_479/BiasAdd/ReadVariableOp�
dense_479/BiasAddAdddense_479/Tensordot:output:0(dense_479/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:��������� �2
dense_479/BiasAdd
tf_op_layer_Pow_59/Pow_59/xConst*
_output_shapes
: *
dtype0*
valueB
 *   A2
tf_op_layer_Pow_59/Pow_59/x�
tf_op_layer_Pow_59/Pow_59Pow$tf_op_layer_Pow_59/Pow_59/x:output:0tf_op_layer_Mul_354/Mul_354:z:0*
T0*
_cloned(*,
_output_shapes
:��������� �2
tf_op_layer_Pow_59/Pow_59�
tf_op_layer_Mul_355/Mul_355Muldense_479/BiasAdd:z:0tf_op_layer_Pow_59/Pow_59:z:0*
T0*
_cloned(*,
_output_shapes
:��������� �2
tf_op_layer_Mul_355/Mul_355�
tf_op_layer_Relu_55/Relu_55Relutf_op_layer_Mul_355/Mul_355:z:0*
T0*
_cloned(*,
_output_shapes
:��������� �2
tf_op_layer_Relu_55/Relu_55�
+tf_op_layer_Max_63/Max_63/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
���������2-
+tf_op_layer_Max_63/Max_63/reduction_indices�
tf_op_layer_Max_63/Max_63Maxinputs_24tf_op_layer_Max_63/Max_63/reduction_indices:output:0*
T0*
_cloned(*+
_output_shapes
:��������� *
	keep_dims(2
tf_op_layer_Max_63/Max_63�
tf_op_layer_Mul_356/Mul_356Mul)tf_op_layer_Relu_55/Relu_55:activations:0"tf_op_layer_Max_63/Max_63:output:0*
T0*
_cloned(*,
_output_shapes
:��������� �2
tf_op_layer_Mul_356/Mul_356x
IdentityIdentitytf_op_layer_Mul_356/Mul_356:z:0*
T0*,
_output_shapes
:��������� �2

Identity"
identityIdentity:output:0*y
_input_shapesh
f:���������:��������� :��������� �::::::::::Q M
'
_output_shapes
:���������
"
_user_specified_name
inputs/0:UQ
+
_output_shapes
:��������� 
"
_user_specified_name
inputs/1:VR
,
_output_shapes
:��������� �
"
_user_specified_name
inputs/2:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: 
�
{
Q__inference_tf_op_layer_AddV2_118_layer_call_and_return_conditional_losses_461567

inputs
inputs_1
identityv
	AddV2_118AddV2inputsinputs_1*
T0*
_cloned(*+
_output_shapes
:��������� 2
	AddV2_118e
IdentityIdentityAddV2_118:z:0*
T0*+
_output_shapes
:��������� 2

Identity"
identityIdentity:output:0*A
_input_shapes0
.:��������� :��������� :S O
+
_output_shapes
:��������� 
 
_user_specified_nameinputs:SO
+
_output_shapes
:��������� 
 
_user_specified_nameinputs
�

*__inference_dense_479_layer_call_fn_462839

inputs
unknown
	unknown_0
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*,
_output_shapes
:��������� �*$
_read_only_resource_inputs
*-
config_proto

GPU

CPU2*0J 8*N
fIRG
E__inference_dense_479_layer_call_and_return_conditional_losses_4618632
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*,
_output_shapes
:��������� �2

Identity"
identityIdentity:output:0*3
_input_shapes"
 :��������� �::22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:��������� �
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
�
{
O__inference_tf_op_layer_Mul_355_layer_call_and_return_conditional_losses_462856
inputs_0
inputs_1
identitys
Mul_355Mulinputs_0inputs_1*
T0*
_cloned(*,
_output_shapes
:��������� �2	
Mul_355d
IdentityIdentityMul_355:z:0*
T0*,
_output_shapes
:��������� �2

Identity"
identityIdentity:output:0*C
_input_shapes2
0:��������� �:��������� �:V R
,
_output_shapes
:��������� �
"
_user_specified_name
inputs/0:VR
,
_output_shapes
:��������� �
"
_user_specified_name
inputs/1
�
w
K__inference_concatenate_179_layer_call_and_return_conditional_losses_462591
inputs_0
inputs_1
identity\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axis�
concatConcatV2inputs_0inputs_1concat/axis:output:0*
N*
T0*+
_output_shapes
:��������� 2
concatg
IdentityIdentityconcat:output:0*
T0*+
_output_shapes
:��������� 2

Identity"
identityIdentity:output:0*A
_input_shapes0
.:��������� :��������� :U Q
+
_output_shapes
:��������� 
"
_user_specified_name
inputs/0:UQ
+
_output_shapes
:��������� 
"
_user_specified_name
inputs/1
�
p
*__inference_color_law_layer_call_fn_462684

inputs
unknown
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*,
_output_shapes
:��������� �*#
_read_only_resource_inputs
*-
config_proto

GPU

CPU2*0J 8*N
fIRG
E__inference_color_law_layer_call_and_return_conditional_losses_4616352
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*,
_output_shapes
:��������� �2

Identity"
identityIdentity:output:0*.
_input_shapes
:��������� :22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:��������� 
 
_user_specified_nameinputs:

_output_shapes
: 
�
Z
>__inference_tf_op_layer_strided_slice_480_layer_call_fn_462559

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*+
_output_shapes
:��������� * 
_read_only_resource_inputs
 *-
config_proto

GPU

CPU2*0J 8*b
f]R[
Y__inference_tf_op_layer_strided_slice_480_layer_call_and_return_conditional_losses_4615372
PartitionedCallp
IdentityIdentityPartitionedCall:output:0*
T0*+
_output_shapes
:��������� 2

Identity"
identityIdentity:output:0**
_input_shapes
:��������� :S O
+
_output_shapes
:��������� 
 
_user_specified_nameinputs
�
�
$__inference_signature_wrapper_462158
conditional_params
	input_240
latent_params
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCalllatent_paramsconditional_params	input_240unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7*
Tin
2*
Tout
2*,
_output_shapes
:��������� �*+
_read_only_resource_inputs
		
*-
config_proto

GPU

CPU2*0J 8**
f%R#
!__inference__wrapped_model_4615072
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*,
_output_shapes
:��������� �2

Identity"
identityIdentity:output:0*y
_input_shapesh
f:��������� :��������� �:���������:::::::::22
StatefulPartitionedCallStatefulPartitionedCall:_ [
+
_output_shapes
:��������� 
,
_user_specified_nameconditional_params:WS
,
_output_shapes
:��������� �
#
_user_specified_name	input_240:VR
'
_output_shapes
:���������
'
_user_specified_namelatent_params:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: 
�
�
*__inference_model_119_layer_call_fn_462131
latent_params
conditional_params
	input_240
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCalllatent_paramsconditional_params	input_240unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7*
Tin
2*
Tout
2*,
_output_shapes
:��������� �*+
_read_only_resource_inputs
		
*-
config_proto

GPU

CPU2*0J 8*N
fIRG
E__inference_model_119_layer_call_and_return_conditional_losses_4621102
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*,
_output_shapes
:��������� �2

Identity"
identityIdentity:output:0*y
_input_shapesh
f:���������:��������� :��������� �:::::::::22
StatefulPartitionedCallStatefulPartitionedCall:V R
'
_output_shapes
:���������
'
_user_specified_namelatent_params:_[
+
_output_shapes
:��������� 
,
_user_specified_nameconditional_params:WS
,
_output_shapes
:��������� �
#
_user_specified_name	input_240:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: 
�
u
Y__inference_tf_op_layer_strided_slice_482_layer_call_and_return_conditional_losses_461584

inputs
identity�
strided_slice_482/beginConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_482/begin
strided_slice_482/endConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_482/end�
strided_slice_482/stridesConst*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_482/strides�
strided_slice_482StridedSliceinputs strided_slice_482/begin:output:0strided_slice_482/end:output:0"strided_slice_482/strides:output:0*
Index0*
T0*
_cloned(*+
_output_shapes
:��������� *
ellipsis_mask2
strided_slice_482r
IdentityIdentitystrided_slice_482:output:0*
T0*+
_output_shapes
:��������� 2

Identity"
identityIdentity:output:0**
_input_shapes
:��������� :S O
+
_output_shapes
:��������� 
 
_user_specified_nameinputs
�
�
E__inference_dense_476_layer_call_and_return_conditional_losses_461694

inputs%
!tensordot_readvariableop_resource#
biasadd_readvariableop_resource
identity��
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes

: *
dtype02
Tensordot/ReadVariableOpj
Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
Tensordot/axesq
Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
Tensordot/freeX
Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:2
Tensordot/Shapet
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/GatherV2/axis�
Tensordot/GatherV2GatherV2Tensordot/Shape:output:0Tensordot/free:output:0 Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
Tensordot/GatherV2x
Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/GatherV2_1/axis�
Tensordot/GatherV2_1GatherV2Tensordot/Shape:output:0Tensordot/axes:output:0"Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
Tensordot/GatherV2_1l
Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot/Const�
Tensordot/ProdProdTensordot/GatherV2:output:0Tensordot/Const:output:0*
T0*
_output_shapes
: 2
Tensordot/Prodp
Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot/Const_1�
Tensordot/Prod_1ProdTensordot/GatherV2_1:output:0Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
Tensordot/Prod_1p
Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/concat/axis�
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat�
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
Tensordot/stack�
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*+
_output_shapes
:��������� 2
Tensordot/transpose�
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:������������������2
Tensordot/Reshape�
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� 2
Tensordot/MatMulp
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot/Const_2t
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/concat_1/axis�
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat_1�
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*+
_output_shapes
:���������  2
	Tensordot�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp�
BiasAddAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������  2	
BiasAddW
ReluReluBiasAdd:z:0*
T0*+
_output_shapes
:���������  2
Reluj
IdentityIdentityRelu:activations:0*
T0*+
_output_shapes
:���������  2

Identity"
identityIdentity:output:0*2
_input_shapes!
:��������� :::S O
+
_output_shapes
:��������� 
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
�
\
0__inference_concatenate_179_layer_call_fn_462597
inputs_0
inputs_1
identity�
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*+
_output_shapes
:��������� * 
_read_only_resource_inputs
 *-
config_proto

GPU

CPU2*0J 8*T
fORM
K__inference_concatenate_179_layer_call_and_return_conditional_losses_4615992
PartitionedCallp
IdentityIdentityPartitionedCall:output:0*
T0*+
_output_shapes
:��������� 2

Identity"
identityIdentity:output:0*A
_input_shapes0
.:��������� :��������� :U Q
+
_output_shapes
:��������� 
"
_user_specified_name
inputs/0:UQ
+
_output_shapes
:��������� 
"
_user_specified_name
inputs/1
�
k
O__inference_tf_op_layer_Relu_55_layer_call_and_return_conditional_losses_461913

inputs
identityh
Relu_55Reluinputs*
T0*
_cloned(*,
_output_shapes
:��������� �2	
Relu_55n
IdentityIdentityRelu_55:activations:0*
T0*,
_output_shapes
:��������� �2

Identity"
identityIdentity:output:0*+
_input_shapes
:��������� �:T P
,
_output_shapes
:��������� �
 
_user_specified_nameinputs
�
P
4__inference_tf_op_layer_Relu_55_layer_call_fn_462872

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*,
_output_shapes
:��������� �* 
_read_only_resource_inputs
 *-
config_proto

GPU

CPU2*0J 8*X
fSRQ
O__inference_tf_op_layer_Relu_55_layer_call_and_return_conditional_losses_4619132
PartitionedCallq
IdentityIdentityPartitionedCall:output:0*
T0*,
_output_shapes
:��������� �2

Identity"
identityIdentity:output:0*+
_input_shapes
:��������� �:T P
,
_output_shapes
:��������� �
 
_user_specified_nameinputs
�
{
O__inference_tf_op_layer_Mul_356_layer_call_and_return_conditional_losses_462889
inputs_0
inputs_1
identitys
Mul_356Mulinputs_0inputs_1*
T0*
_cloned(*,
_output_shapes
:��������� �2	
Mul_356d
IdentityIdentityMul_356:z:0*
T0*,
_output_shapes
:��������� �2

Identity"
identityIdentity:output:0*B
_input_shapes1
/:��������� �:��������� :V R
,
_output_shapes
:��������� �
"
_user_specified_name
inputs/0:UQ
+
_output_shapes
:��������� 
"
_user_specified_name
inputs/1
�
Z
>__inference_tf_op_layer_strided_slice_482_layer_call_fn_462610

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*+
_output_shapes
:��������� * 
_read_only_resource_inputs
 *-
config_proto

GPU

CPU2*0J 8*b
f]R[
Y__inference_tf_op_layer_strided_slice_482_layer_call_and_return_conditional_losses_4615842
PartitionedCallp
IdentityIdentityPartitionedCall:output:0*
T0*+
_output_shapes
:��������� 2

Identity"
identityIdentity:output:0**
_input_shapes
:��������� :S O
+
_output_shapes
:��������� 
 
_user_specified_nameinputs
�
O
3__inference_tf_op_layer_Max_63_layer_call_fn_462883

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*+
_output_shapes
:��������� * 
_read_only_resource_inputs
 *-
config_proto

GPU

CPU2*0J 8*W
fRRP
N__inference_tf_op_layer_Max_63_layer_call_and_return_conditional_losses_4619272
PartitionedCallp
IdentityIdentityPartitionedCall:output:0*
T0*+
_output_shapes
:��������� 2

Identity"
identityIdentity:output:0*+
_input_shapes
:��������� �:T P
,
_output_shapes
:��������� �
 
_user_specified_nameinputs
�
j
N__inference_tf_op_layer_Max_63_layer_call_and_return_conditional_losses_461927

inputs
identity
Max_63/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
���������2
Max_63/reduction_indices�
Max_63Maxinputs!Max_63/reduction_indices:output:0*
T0*
_cloned(*+
_output_shapes
:��������� *
	keep_dims(2
Max_63g
IdentityIdentityMax_63:output:0*
T0*+
_output_shapes
:��������� 2

Identity"
identityIdentity:output:0*+
_input_shapes
:��������� �:T P
,
_output_shapes
:��������� �
 
_user_specified_nameinputs
�
�
E__inference_dense_477_layer_call_and_return_conditional_losses_462728

inputs%
!tensordot_readvariableop_resource#
biasadd_readvariableop_resource
identity��
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes
:	 �*
dtype02
Tensordot/ReadVariableOpj
Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
Tensordot/axesq
Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
Tensordot/freeX
Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:2
Tensordot/Shapet
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/GatherV2/axis�
Tensordot/GatherV2GatherV2Tensordot/Shape:output:0Tensordot/free:output:0 Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
Tensordot/GatherV2x
Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/GatherV2_1/axis�
Tensordot/GatherV2_1GatherV2Tensordot/Shape:output:0Tensordot/axes:output:0"Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
Tensordot/GatherV2_1l
Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot/Const�
Tensordot/ProdProdTensordot/GatherV2:output:0Tensordot/Const:output:0*
T0*
_output_shapes
: 2
Tensordot/Prodp
Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot/Const_1�
Tensordot/Prod_1ProdTensordot/GatherV2_1:output:0Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
Tensordot/Prod_1p
Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/concat/axis�
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat�
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
Tensordot/stack�
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*+
_output_shapes
:���������  2
Tensordot/transpose�
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:������������������2
Tensordot/Reshape�
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
Tensordot/MatMulq
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:�2
Tensordot/Const_2t
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/concat_1/axis�
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat_1�
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*,
_output_shapes
:��������� �2
	Tensordot�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02
BiasAdd/ReadVariableOp�
BiasAddAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:��������� �2	
BiasAddX
ReluReluBiasAdd:z:0*
T0*,
_output_shapes
:��������� �2
Reluk
IdentityIdentityRelu:activations:0*
T0*,
_output_shapes
:��������� �2

Identity"
identityIdentity:output:0*2
_input_shapes!
:���������  :::S O
+
_output_shapes
:���������  
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
�
�
E__inference_color_law_layer_call_and_return_conditional_losses_461635

inputs%
!tensordot_readvariableop_resource
identity��
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes
:	�*
dtype02
Tensordot/ReadVariableOpj
Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
Tensordot/axesq
Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
Tensordot/freeX
Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:2
Tensordot/Shapet
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/GatherV2/axis�
Tensordot/GatherV2GatherV2Tensordot/Shape:output:0Tensordot/free:output:0 Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
Tensordot/GatherV2x
Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/GatherV2_1/axis�
Tensordot/GatherV2_1GatherV2Tensordot/Shape:output:0Tensordot/axes:output:0"Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
Tensordot/GatherV2_1l
Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot/Const�
Tensordot/ProdProdTensordot/GatherV2:output:0Tensordot/Const:output:0*
T0*
_output_shapes
: 2
Tensordot/Prodp
Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot/Const_1�
Tensordot/Prod_1ProdTensordot/GatherV2_1:output:0Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
Tensordot/Prod_1p
Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/concat/axis�
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat�
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
Tensordot/stack�
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*+
_output_shapes
:��������� 2
Tensordot/transpose�
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:������������������2
Tensordot/Reshape�
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
Tensordot/MatMulq
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:�2
Tensordot/Const_2t
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/concat_1/axis�
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat_1�
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*,
_output_shapes
:��������� �2
	Tensordotk
IdentityIdentityTensordot:output:0*
T0*,
_output_shapes
:��������� �2

Identity"
identityIdentity:output:0*.
_input_shapes
:��������� ::S O
+
_output_shapes
:��������� 
 
_user_specified_nameinputs:

_output_shapes
: 
�
}
Q__inference_tf_op_layer_AddV2_119_layer_call_and_return_conditional_losses_462743
inputs_0
inputs_1
identityy
	AddV2_119AddV2inputs_0inputs_1*
T0*
_cloned(*,
_output_shapes
:��������� �2
	AddV2_119f
IdentityIdentityAddV2_119:z:0*
T0*,
_output_shapes
:��������� �2

Identity"
identityIdentity:output:0*B
_input_shapes1
/:��������� �:��������� :V R
,
_output_shapes
:��������� �
"
_user_specified_name
inputs/0:UQ
+
_output_shapes
:��������� 
"
_user_specified_name
inputs/1
�
u
Y__inference_tf_op_layer_strided_slice_483_layer_call_and_return_conditional_losses_461553

inputs
identity�
strided_slice_483/beginConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_483/begin
strided_slice_483/endConst*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice_483/end�
strided_slice_483/stridesConst*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_483/strides�
strided_slice_483StridedSliceinputs strided_slice_483/begin:output:0strided_slice_483/end:output:0"strided_slice_483/strides:output:0*
Index0*
T0*
_cloned(*+
_output_shapes
:��������� *
ellipsis_mask*
end_mask2
strided_slice_483r
IdentityIdentitystrided_slice_483:output:0*
T0*+
_output_shapes
:��������� 2

Identity"
identityIdentity:output:0**
_input_shapes
:��������� :S O
+
_output_shapes
:��������� 
 
_user_specified_nameinputs
�.
�
"__inference__traced_restore_462990
file_prefix%
!assignvariableop_dense_476_kernel%
!assignvariableop_1_dense_476_bias*
&assignvariableop_2_color_law_62_kernel'
#assignvariableop_3_dense_477_kernel%
!assignvariableop_4_dense_477_bias'
#assignvariableop_5_dense_478_kernel%
!assignvariableop_6_dense_478_bias'
#assignvariableop_7_dense_479_kernel%
!assignvariableop_8_dense_479_bias
identity_10��AssignVariableOp�AssignVariableOp_1�AssignVariableOp_2�AssignVariableOp_3�AssignVariableOp_4�AssignVariableOp_5�AssignVariableOp_6�AssignVariableOp_7�AssignVariableOp_8�	RestoreV2�RestoreV2_1�
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:	*
dtype0*�
value�B�	B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE2
RestoreV2/tensor_names�
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:	*
dtype0*%
valueB	B B B B B B B B B 2
RestoreV2/shape_and_slices�
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*8
_output_shapes&
$:::::::::*
dtypes
2	2
	RestoreV2X
IdentityIdentityRestoreV2:tensors:0*
T0*
_output_shapes
:2

Identity�
AssignVariableOpAssignVariableOp!assignvariableop_dense_476_kernelIdentity:output:0*
_output_shapes
 *
dtype02
AssignVariableOp\

Identity_1IdentityRestoreV2:tensors:1*
T0*
_output_shapes
:2

Identity_1�
AssignVariableOp_1AssignVariableOp!assignvariableop_1_dense_476_biasIdentity_1:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_1\

Identity_2IdentityRestoreV2:tensors:2*
T0*
_output_shapes
:2

Identity_2�
AssignVariableOp_2AssignVariableOp&assignvariableop_2_color_law_62_kernelIdentity_2:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_2\

Identity_3IdentityRestoreV2:tensors:3*
T0*
_output_shapes
:2

Identity_3�
AssignVariableOp_3AssignVariableOp#assignvariableop_3_dense_477_kernelIdentity_3:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_3\

Identity_4IdentityRestoreV2:tensors:4*
T0*
_output_shapes
:2

Identity_4�
AssignVariableOp_4AssignVariableOp!assignvariableop_4_dense_477_biasIdentity_4:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_4\

Identity_5IdentityRestoreV2:tensors:5*
T0*
_output_shapes
:2

Identity_5�
AssignVariableOp_5AssignVariableOp#assignvariableop_5_dense_478_kernelIdentity_5:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_5\

Identity_6IdentityRestoreV2:tensors:6*
T0*
_output_shapes
:2

Identity_6�
AssignVariableOp_6AssignVariableOp!assignvariableop_6_dense_478_biasIdentity_6:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_6\

Identity_7IdentityRestoreV2:tensors:7*
T0*
_output_shapes
:2

Identity_7�
AssignVariableOp_7AssignVariableOp#assignvariableop_7_dense_479_kernelIdentity_7:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_7\

Identity_8IdentityRestoreV2:tensors:8*
T0*
_output_shapes
:2

Identity_8�
AssignVariableOp_8AssignVariableOp!assignvariableop_8_dense_479_biasIdentity_8:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_8�
RestoreV2_1/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*1
value(B&B_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2_1/tensor_names�
RestoreV2_1/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueB
B 2
RestoreV2_1/shape_and_slices�
RestoreV2_1	RestoreV2file_prefix!RestoreV2_1/tensor_names:output:0%RestoreV2_1/shape_and_slices:output:0
^RestoreV2"/device:CPU:0*
_output_shapes
:*
dtypes
22
RestoreV2_19
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp�

Identity_9Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2

Identity_9�
Identity_10IdentityIdentity_9:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8
^RestoreV2^RestoreV2_1*
T0*
_output_shapes
: 2
Identity_10"#
identity_10Identity_10:output:0*9
_input_shapes(
&: :::::::::2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12(
AssignVariableOp_2AssignVariableOp_22(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82
	RestoreV2	RestoreV22
RestoreV2_1RestoreV2_1:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: 
�
j
N__inference_tf_op_layer_Pow_59_layer_call_and_return_conditional_losses_462845

inputs
identityY
Pow_59/xConst*
_output_shapes
: *
dtype0*
valueB
 *   A2

Pow_59/xx
Pow_59PowPow_59/x:output:0inputs*
T0*
_cloned(*,
_output_shapes
:��������� �2
Pow_59c
IdentityIdentity
Pow_59:z:0*
T0*,
_output_shapes
:��������� �2

Identity"
identityIdentity:output:0*+
_input_shapes
:��������� �:T P
,
_output_shapes
:��������� �
 
_user_specified_nameinputs
�
u
K__inference_concatenate_179_layer_call_and_return_conditional_losses_461599

inputs
inputs_1
identity\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axis�
concatConcatV2inputsinputs_1concat/axis:output:0*
N*
T0*+
_output_shapes
:��������� 2
concatg
IdentityIdentityconcat:output:0*
T0*+
_output_shapes
:��������� 2

Identity"
identityIdentity:output:0*A
_input_shapes0
.:��������� :��������� :S O
+
_output_shapes
:��������� 
 
_user_specified_nameinputs:SO
+
_output_shapes
:��������� 
 
_user_specified_nameinputs
�
u
Y__inference_tf_op_layer_strided_slice_482_layer_call_and_return_conditional_losses_462605

inputs
identity�
strided_slice_482/beginConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_482/begin
strided_slice_482/endConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_482/end�
strided_slice_482/stridesConst*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_482/strides�
strided_slice_482StridedSliceinputs strided_slice_482/begin:output:0strided_slice_482/end:output:0"strided_slice_482/strides:output:0*
Index0*
T0*
_cloned(*+
_output_shapes
:��������� *
ellipsis_mask2
strided_slice_482r
IdentityIdentitystrided_slice_482:output:0*
T0*+
_output_shapes
:��������� 2

Identity"
identityIdentity:output:0**
_input_shapes
:��������� :S O
+
_output_shapes
:��������� 
 
_user_specified_nameinputs
�
�
E__inference_dense_479_layer_call_and_return_conditional_losses_461863

inputs%
!tensordot_readvariableop_resource#
biasadd_readvariableop_resource
identity��
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource* 
_output_shapes
:
��*
dtype02
Tensordot/ReadVariableOpj
Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
Tensordot/axesq
Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
Tensordot/freeX
Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:2
Tensordot/Shapet
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/GatherV2/axis�
Tensordot/GatherV2GatherV2Tensordot/Shape:output:0Tensordot/free:output:0 Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
Tensordot/GatherV2x
Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/GatherV2_1/axis�
Tensordot/GatherV2_1GatherV2Tensordot/Shape:output:0Tensordot/axes:output:0"Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
Tensordot/GatherV2_1l
Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot/Const�
Tensordot/ProdProdTensordot/GatherV2:output:0Tensordot/Const:output:0*
T0*
_output_shapes
: 2
Tensordot/Prodp
Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot/Const_1�
Tensordot/Prod_1ProdTensordot/GatherV2_1:output:0Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
Tensordot/Prod_1p
Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/concat/axis�
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat�
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
Tensordot/stack�
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*,
_output_shapes
:��������� �2
Tensordot/transpose�
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:������������������2
Tensordot/Reshape�
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
Tensordot/MatMulq
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:�2
Tensordot/Const_2t
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/concat_1/axis�
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat_1�
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*,
_output_shapes
:��������� �2
	Tensordot�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02
BiasAdd/ReadVariableOp�
BiasAddAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:��������� �2	
BiasAddd
IdentityIdentityBiasAdd:z:0*
T0*,
_output_shapes
:��������� �2

Identity"
identityIdentity:output:0*3
_input_shapes"
 :��������� �:::T P
,
_output_shapes
:��������� �
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
�
y
O__inference_tf_op_layer_Mul_356_layer_call_and_return_conditional_losses_461941

inputs
inputs_1
identityq
Mul_356Mulinputsinputs_1*
T0*
_cloned(*,
_output_shapes
:��������� �2	
Mul_356d
IdentityIdentityMul_356:z:0*
T0*,
_output_shapes
:��������� �2

Identity"
identityIdentity:output:0*B
_input_shapes1
/:��������� �:��������� :T P
,
_output_shapes
:��������� �
 
_user_specified_nameinputs:SO
+
_output_shapes
:��������� 
 
_user_specified_nameinputs
�
k
O__inference_tf_op_layer_Mul_354_layer_call_and_return_conditional_losses_461778

inputs
identity[
	Mul_354/xConst*
_output_shapes
: *
dtype0*
valueB
 *��̾2
	Mul_354/x{
Mul_354MulMul_354/x:output:0inputs*
T0*
_cloned(*,
_output_shapes
:��������� �2	
Mul_354d
IdentityIdentityMul_354:z:0*
T0*,
_output_shapes
:��������� �2

Identity"
identityIdentity:output:0*+
_input_shapes
:��������� �:T P
,
_output_shapes
:��������� �
 
_user_specified_nameinputs
��
�
E__inference_model_119_layer_call_and_return_conditional_losses_462496
inputs_0
inputs_1
inputs_2/
+color_law_tensordot_readvariableop_resource/
+dense_476_tensordot_readvariableop_resource-
)dense_476_biasadd_readvariableop_resource/
+dense_477_tensordot_readvariableop_resource-
)dense_477_biasadd_readvariableop_resource/
+dense_478_tensordot_readvariableop_resource-
)dense_478_biasadd_readvariableop_resource/
+dense_479_tensordot_readvariableop_resource-
)dense_479_biasadd_readvariableop_resource
identity��
repeat_vector_59/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2!
repeat_vector_59/ExpandDims/dim�
repeat_vector_59/ExpandDims
ExpandDimsinputs_0(repeat_vector_59/ExpandDims/dim:output:0*
T0*+
_output_shapes
:���������2
repeat_vector_59/ExpandDims�
repeat_vector_59/stackConst*
_output_shapes
:*
dtype0*!
valueB"          2
repeat_vector_59/stack�
repeat_vector_59/TileTile$repeat_vector_59/ExpandDims:output:0repeat_vector_59/stack:output:0*
T0*+
_output_shapes
:��������� 2
repeat_vector_59/Tile�
5tf_op_layer_strided_slice_480/strided_slice_480/beginConst*
_output_shapes
:*
dtype0*
valueB"        27
5tf_op_layer_strided_slice_480/strided_slice_480/begin�
3tf_op_layer_strided_slice_480/strided_slice_480/endConst*
_output_shapes
:*
dtype0*
valueB"       25
3tf_op_layer_strided_slice_480/strided_slice_480/end�
7tf_op_layer_strided_slice_480/strided_slice_480/stridesConst*
_output_shapes
:*
dtype0*
valueB"      29
7tf_op_layer_strided_slice_480/strided_slice_480/strides�
/tf_op_layer_strided_slice_480/strided_slice_480StridedSlicerepeat_vector_59/Tile:output:0>tf_op_layer_strided_slice_480/strided_slice_480/begin:output:0<tf_op_layer_strided_slice_480/strided_slice_480/end:output:0@tf_op_layer_strided_slice_480/strided_slice_480/strides:output:0*
Index0*
T0*
_cloned(*+
_output_shapes
:��������� *
ellipsis_mask21
/tf_op_layer_strided_slice_480/strided_slice_480�
5tf_op_layer_strided_slice_483/strided_slice_483/beginConst*
_output_shapes
:*
dtype0*
valueB"       27
5tf_op_layer_strided_slice_483/strided_slice_483/begin�
3tf_op_layer_strided_slice_483/strided_slice_483/endConst*
_output_shapes
:*
dtype0*
valueB"        25
3tf_op_layer_strided_slice_483/strided_slice_483/end�
7tf_op_layer_strided_slice_483/strided_slice_483/stridesConst*
_output_shapes
:*
dtype0*
valueB"      29
7tf_op_layer_strided_slice_483/strided_slice_483/strides�
/tf_op_layer_strided_slice_483/strided_slice_483StridedSlicerepeat_vector_59/Tile:output:0>tf_op_layer_strided_slice_483/strided_slice_483/begin:output:0<tf_op_layer_strided_slice_483/strided_slice_483/end:output:0@tf_op_layer_strided_slice_483/strided_slice_483/strides:output:0*
Index0*
T0*
_cloned(*+
_output_shapes
:��������� *
ellipsis_mask*
end_mask21
/tf_op_layer_strided_slice_483/strided_slice_483�
tf_op_layer_AddV2_118/AddV2_118AddV2inputs_18tf_op_layer_strided_slice_480/strided_slice_480:output:0*
T0*
_cloned(*+
_output_shapes
:��������� 2!
tf_op_layer_AddV2_118/AddV2_118�
5tf_op_layer_strided_slice_482/strided_slice_482/beginConst*
_output_shapes
:*
dtype0*
valueB"       27
5tf_op_layer_strided_slice_482/strided_slice_482/begin�
3tf_op_layer_strided_slice_482/strided_slice_482/endConst*
_output_shapes
:*
dtype0*
valueB"       25
3tf_op_layer_strided_slice_482/strided_slice_482/end�
7tf_op_layer_strided_slice_482/strided_slice_482/stridesConst*
_output_shapes
:*
dtype0*
valueB"      29
7tf_op_layer_strided_slice_482/strided_slice_482/strides�
/tf_op_layer_strided_slice_482/strided_slice_482StridedSlicerepeat_vector_59/Tile:output:0>tf_op_layer_strided_slice_482/strided_slice_482/begin:output:0<tf_op_layer_strided_slice_482/strided_slice_482/end:output:0@tf_op_layer_strided_slice_482/strided_slice_482/strides:output:0*
Index0*
T0*
_cloned(*+
_output_shapes
:��������� *
ellipsis_mask21
/tf_op_layer_strided_slice_482/strided_slice_482|
concatenate_179/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concatenate_179/concat/axis�
concatenate_179/concatConcatV28tf_op_layer_strided_slice_483/strided_slice_483:output:0#tf_op_layer_AddV2_118/AddV2_118:z:0$concatenate_179/concat/axis:output:0*
N*
T0*+
_output_shapes
:��������� 2
concatenate_179/concat�
"color_law/Tensordot/ReadVariableOpReadVariableOp+color_law_tensordot_readvariableop_resource*
_output_shapes
:	�*
dtype02$
"color_law/Tensordot/ReadVariableOp~
color_law/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
color_law/Tensordot/axes�
color_law/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
color_law/Tensordot/free�
color_law/Tensordot/ShapeShape8tf_op_layer_strided_slice_482/strided_slice_482:output:0*
T0*
_output_shapes
:2
color_law/Tensordot/Shape�
!color_law/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2#
!color_law/Tensordot/GatherV2/axis�
color_law/Tensordot/GatherV2GatherV2"color_law/Tensordot/Shape:output:0!color_law/Tensordot/free:output:0*color_law/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
color_law/Tensordot/GatherV2�
#color_law/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2%
#color_law/Tensordot/GatherV2_1/axis�
color_law/Tensordot/GatherV2_1GatherV2"color_law/Tensordot/Shape:output:0!color_law/Tensordot/axes:output:0,color_law/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2 
color_law/Tensordot/GatherV2_1�
color_law/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
color_law/Tensordot/Const�
color_law/Tensordot/ProdProd%color_law/Tensordot/GatherV2:output:0"color_law/Tensordot/Const:output:0*
T0*
_output_shapes
: 2
color_law/Tensordot/Prod�
color_law/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
color_law/Tensordot/Const_1�
color_law/Tensordot/Prod_1Prod'color_law/Tensordot/GatherV2_1:output:0$color_law/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
color_law/Tensordot/Prod_1�
color_law/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2!
color_law/Tensordot/concat/axis�
color_law/Tensordot/concatConcatV2!color_law/Tensordot/free:output:0!color_law/Tensordot/axes:output:0(color_law/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
color_law/Tensordot/concat�
color_law/Tensordot/stackPack!color_law/Tensordot/Prod:output:0#color_law/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
color_law/Tensordot/stack�
color_law/Tensordot/transpose	Transpose8tf_op_layer_strided_slice_482/strided_slice_482:output:0#color_law/Tensordot/concat:output:0*
T0*+
_output_shapes
:��������� 2
color_law/Tensordot/transpose�
color_law/Tensordot/ReshapeReshape!color_law/Tensordot/transpose:y:0"color_law/Tensordot/stack:output:0*
T0*0
_output_shapes
:������������������2
color_law/Tensordot/Reshape�
color_law/Tensordot/MatMulMatMul$color_law/Tensordot/Reshape:output:0*color_law/Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
color_law/Tensordot/MatMul�
color_law/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:�2
color_law/Tensordot/Const_2�
!color_law/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2#
!color_law/Tensordot/concat_1/axis�
color_law/Tensordot/concat_1ConcatV2%color_law/Tensordot/GatherV2:output:0$color_law/Tensordot/Const_2:output:0*color_law/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
color_law/Tensordot/concat_1�
color_law/TensordotReshape$color_law/Tensordot/MatMul:product:0%color_law/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:��������� �2
color_law/Tensordot�
5tf_op_layer_strided_slice_481/strided_slice_481/beginConst*
_output_shapes
:*
dtype0*
valueB"       27
5tf_op_layer_strided_slice_481/strided_slice_481/begin�
3tf_op_layer_strided_slice_481/strided_slice_481/endConst*
_output_shapes
:*
dtype0*
valueB"       25
3tf_op_layer_strided_slice_481/strided_slice_481/end�
7tf_op_layer_strided_slice_481/strided_slice_481/stridesConst*
_output_shapes
:*
dtype0*
valueB"      29
7tf_op_layer_strided_slice_481/strided_slice_481/strides�
/tf_op_layer_strided_slice_481/strided_slice_481StridedSlicerepeat_vector_59/Tile:output:0>tf_op_layer_strided_slice_481/strided_slice_481/begin:output:0<tf_op_layer_strided_slice_481/strided_slice_481/end:output:0@tf_op_layer_strided_slice_481/strided_slice_481/strides:output:0*
Index0*
T0*
_cloned(*+
_output_shapes
:��������� *
ellipsis_mask21
/tf_op_layer_strided_slice_481/strided_slice_481�
"dense_476/Tensordot/ReadVariableOpReadVariableOp+dense_476_tensordot_readvariableop_resource*
_output_shapes

: *
dtype02$
"dense_476/Tensordot/ReadVariableOp~
dense_476/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
dense_476/Tensordot/axes�
dense_476/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
dense_476/Tensordot/free�
dense_476/Tensordot/ShapeShapeconcatenate_179/concat:output:0*
T0*
_output_shapes
:2
dense_476/Tensordot/Shape�
!dense_476/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2#
!dense_476/Tensordot/GatherV2/axis�
dense_476/Tensordot/GatherV2GatherV2"dense_476/Tensordot/Shape:output:0!dense_476/Tensordot/free:output:0*dense_476/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_476/Tensordot/GatherV2�
#dense_476/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2%
#dense_476/Tensordot/GatherV2_1/axis�
dense_476/Tensordot/GatherV2_1GatherV2"dense_476/Tensordot/Shape:output:0!dense_476/Tensordot/axes:output:0,dense_476/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2 
dense_476/Tensordot/GatherV2_1�
dense_476/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
dense_476/Tensordot/Const�
dense_476/Tensordot/ProdProd%dense_476/Tensordot/GatherV2:output:0"dense_476/Tensordot/Const:output:0*
T0*
_output_shapes
: 2
dense_476/Tensordot/Prod�
dense_476/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
dense_476/Tensordot/Const_1�
dense_476/Tensordot/Prod_1Prod'dense_476/Tensordot/GatherV2_1:output:0$dense_476/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
dense_476/Tensordot/Prod_1�
dense_476/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2!
dense_476/Tensordot/concat/axis�
dense_476/Tensordot/concatConcatV2!dense_476/Tensordot/free:output:0!dense_476/Tensordot/axes:output:0(dense_476/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
dense_476/Tensordot/concat�
dense_476/Tensordot/stackPack!dense_476/Tensordot/Prod:output:0#dense_476/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
dense_476/Tensordot/stack�
dense_476/Tensordot/transpose	Transposeconcatenate_179/concat:output:0#dense_476/Tensordot/concat:output:0*
T0*+
_output_shapes
:��������� 2
dense_476/Tensordot/transpose�
dense_476/Tensordot/ReshapeReshape!dense_476/Tensordot/transpose:y:0"dense_476/Tensordot/stack:output:0*
T0*0
_output_shapes
:������������������2
dense_476/Tensordot/Reshape�
dense_476/Tensordot/MatMulMatMul$dense_476/Tensordot/Reshape:output:0*dense_476/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� 2
dense_476/Tensordot/MatMul�
dense_476/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: 2
dense_476/Tensordot/Const_2�
!dense_476/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2#
!dense_476/Tensordot/concat_1/axis�
dense_476/Tensordot/concat_1ConcatV2%dense_476/Tensordot/GatherV2:output:0$dense_476/Tensordot/Const_2:output:0*dense_476/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
dense_476/Tensordot/concat_1�
dense_476/TensordotReshape$dense_476/Tensordot/MatMul:product:0%dense_476/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:���������  2
dense_476/Tensordot�
 dense_476/BiasAdd/ReadVariableOpReadVariableOp)dense_476_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02"
 dense_476/BiasAdd/ReadVariableOp�
dense_476/BiasAddAdddense_476/Tensordot:output:0(dense_476/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������  2
dense_476/BiasAddu
dense_476/ReluReludense_476/BiasAdd:z:0*
T0*+
_output_shapes
:���������  2
dense_476/Relu�
tf_op_layer_AddV2_119/AddV2_119AddV2color_law/Tensordot:output:08tf_op_layer_strided_slice_481/strided_slice_481:output:0*
T0*
_cloned(*,
_output_shapes
:��������� �2!
tf_op_layer_AddV2_119/AddV2_119�
"dense_477/Tensordot/ReadVariableOpReadVariableOp+dense_477_tensordot_readvariableop_resource*
_output_shapes
:	 �*
dtype02$
"dense_477/Tensordot/ReadVariableOp~
dense_477/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
dense_477/Tensordot/axes�
dense_477/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
dense_477/Tensordot/free�
dense_477/Tensordot/ShapeShapedense_476/Relu:activations:0*
T0*
_output_shapes
:2
dense_477/Tensordot/Shape�
!dense_477/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2#
!dense_477/Tensordot/GatherV2/axis�
dense_477/Tensordot/GatherV2GatherV2"dense_477/Tensordot/Shape:output:0!dense_477/Tensordot/free:output:0*dense_477/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_477/Tensordot/GatherV2�
#dense_477/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2%
#dense_477/Tensordot/GatherV2_1/axis�
dense_477/Tensordot/GatherV2_1GatherV2"dense_477/Tensordot/Shape:output:0!dense_477/Tensordot/axes:output:0,dense_477/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2 
dense_477/Tensordot/GatherV2_1�
dense_477/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
dense_477/Tensordot/Const�
dense_477/Tensordot/ProdProd%dense_477/Tensordot/GatherV2:output:0"dense_477/Tensordot/Const:output:0*
T0*
_output_shapes
: 2
dense_477/Tensordot/Prod�
dense_477/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
dense_477/Tensordot/Const_1�
dense_477/Tensordot/Prod_1Prod'dense_477/Tensordot/GatherV2_1:output:0$dense_477/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
dense_477/Tensordot/Prod_1�
dense_477/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2!
dense_477/Tensordot/concat/axis�
dense_477/Tensordot/concatConcatV2!dense_477/Tensordot/free:output:0!dense_477/Tensordot/axes:output:0(dense_477/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
dense_477/Tensordot/concat�
dense_477/Tensordot/stackPack!dense_477/Tensordot/Prod:output:0#dense_477/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
dense_477/Tensordot/stack�
dense_477/Tensordot/transpose	Transposedense_476/Relu:activations:0#dense_477/Tensordot/concat:output:0*
T0*+
_output_shapes
:���������  2
dense_477/Tensordot/transpose�
dense_477/Tensordot/ReshapeReshape!dense_477/Tensordot/transpose:y:0"dense_477/Tensordot/stack:output:0*
T0*0
_output_shapes
:������������������2
dense_477/Tensordot/Reshape�
dense_477/Tensordot/MatMulMatMul$dense_477/Tensordot/Reshape:output:0*dense_477/Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
dense_477/Tensordot/MatMul�
dense_477/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:�2
dense_477/Tensordot/Const_2�
!dense_477/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2#
!dense_477/Tensordot/concat_1/axis�
dense_477/Tensordot/concat_1ConcatV2%dense_477/Tensordot/GatherV2:output:0$dense_477/Tensordot/Const_2:output:0*dense_477/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
dense_477/Tensordot/concat_1�
dense_477/TensordotReshape$dense_477/Tensordot/MatMul:product:0%dense_477/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:��������� �2
dense_477/Tensordot�
 dense_477/BiasAdd/ReadVariableOpReadVariableOp)dense_477_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02"
 dense_477/BiasAdd/ReadVariableOp�
dense_477/BiasAddAdddense_477/Tensordot:output:0(dense_477/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:��������� �2
dense_477/BiasAddv
dense_477/ReluReludense_477/BiasAdd:z:0*
T0*,
_output_shapes
:��������� �2
dense_477/Relu�
tf_op_layer_Mul_354/Mul_354/xConst*
_output_shapes
: *
dtype0*
valueB
 *��̾2
tf_op_layer_Mul_354/Mul_354/x�
tf_op_layer_Mul_354/Mul_354Mul&tf_op_layer_Mul_354/Mul_354/x:output:0#tf_op_layer_AddV2_119/AddV2_119:z:0*
T0*
_cloned(*,
_output_shapes
:��������� �2
tf_op_layer_Mul_354/Mul_354�
"dense_478/Tensordot/ReadVariableOpReadVariableOp+dense_478_tensordot_readvariableop_resource* 
_output_shapes
:
��*
dtype02$
"dense_478/Tensordot/ReadVariableOp~
dense_478/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
dense_478/Tensordot/axes�
dense_478/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
dense_478/Tensordot/free�
dense_478/Tensordot/ShapeShapedense_477/Relu:activations:0*
T0*
_output_shapes
:2
dense_478/Tensordot/Shape�
!dense_478/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2#
!dense_478/Tensordot/GatherV2/axis�
dense_478/Tensordot/GatherV2GatherV2"dense_478/Tensordot/Shape:output:0!dense_478/Tensordot/free:output:0*dense_478/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_478/Tensordot/GatherV2�
#dense_478/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2%
#dense_478/Tensordot/GatherV2_1/axis�
dense_478/Tensordot/GatherV2_1GatherV2"dense_478/Tensordot/Shape:output:0!dense_478/Tensordot/axes:output:0,dense_478/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2 
dense_478/Tensordot/GatherV2_1�
dense_478/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
dense_478/Tensordot/Const�
dense_478/Tensordot/ProdProd%dense_478/Tensordot/GatherV2:output:0"dense_478/Tensordot/Const:output:0*
T0*
_output_shapes
: 2
dense_478/Tensordot/Prod�
dense_478/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
dense_478/Tensordot/Const_1�
dense_478/Tensordot/Prod_1Prod'dense_478/Tensordot/GatherV2_1:output:0$dense_478/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
dense_478/Tensordot/Prod_1�
dense_478/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2!
dense_478/Tensordot/concat/axis�
dense_478/Tensordot/concatConcatV2!dense_478/Tensordot/free:output:0!dense_478/Tensordot/axes:output:0(dense_478/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
dense_478/Tensordot/concat�
dense_478/Tensordot/stackPack!dense_478/Tensordot/Prod:output:0#dense_478/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
dense_478/Tensordot/stack�
dense_478/Tensordot/transpose	Transposedense_477/Relu:activations:0#dense_478/Tensordot/concat:output:0*
T0*,
_output_shapes
:��������� �2
dense_478/Tensordot/transpose�
dense_478/Tensordot/ReshapeReshape!dense_478/Tensordot/transpose:y:0"dense_478/Tensordot/stack:output:0*
T0*0
_output_shapes
:������������������2
dense_478/Tensordot/Reshape�
dense_478/Tensordot/MatMulMatMul$dense_478/Tensordot/Reshape:output:0*dense_478/Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
dense_478/Tensordot/MatMul�
dense_478/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:�2
dense_478/Tensordot/Const_2�
!dense_478/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2#
!dense_478/Tensordot/concat_1/axis�
dense_478/Tensordot/concat_1ConcatV2%dense_478/Tensordot/GatherV2:output:0$dense_478/Tensordot/Const_2:output:0*dense_478/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
dense_478/Tensordot/concat_1�
dense_478/TensordotReshape$dense_478/Tensordot/MatMul:product:0%dense_478/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:��������� �2
dense_478/Tensordot�
 dense_478/BiasAdd/ReadVariableOpReadVariableOp)dense_478_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02"
 dense_478/BiasAdd/ReadVariableOp�
dense_478/BiasAddAdddense_478/Tensordot:output:0(dense_478/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:��������� �2
dense_478/BiasAddv
dense_478/ReluReludense_478/BiasAdd:z:0*
T0*,
_output_shapes
:��������� �2
dense_478/Relu�
"dense_479/Tensordot/ReadVariableOpReadVariableOp+dense_479_tensordot_readvariableop_resource* 
_output_shapes
:
��*
dtype02$
"dense_479/Tensordot/ReadVariableOp~
dense_479/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
dense_479/Tensordot/axes�
dense_479/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
dense_479/Tensordot/free�
dense_479/Tensordot/ShapeShapedense_478/Relu:activations:0*
T0*
_output_shapes
:2
dense_479/Tensordot/Shape�
!dense_479/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2#
!dense_479/Tensordot/GatherV2/axis�
dense_479/Tensordot/GatherV2GatherV2"dense_479/Tensordot/Shape:output:0!dense_479/Tensordot/free:output:0*dense_479/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_479/Tensordot/GatherV2�
#dense_479/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2%
#dense_479/Tensordot/GatherV2_1/axis�
dense_479/Tensordot/GatherV2_1GatherV2"dense_479/Tensordot/Shape:output:0!dense_479/Tensordot/axes:output:0,dense_479/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2 
dense_479/Tensordot/GatherV2_1�
dense_479/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
dense_479/Tensordot/Const�
dense_479/Tensordot/ProdProd%dense_479/Tensordot/GatherV2:output:0"dense_479/Tensordot/Const:output:0*
T0*
_output_shapes
: 2
dense_479/Tensordot/Prod�
dense_479/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
dense_479/Tensordot/Const_1�
dense_479/Tensordot/Prod_1Prod'dense_479/Tensordot/GatherV2_1:output:0$dense_479/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
dense_479/Tensordot/Prod_1�
dense_479/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2!
dense_479/Tensordot/concat/axis�
dense_479/Tensordot/concatConcatV2!dense_479/Tensordot/free:output:0!dense_479/Tensordot/axes:output:0(dense_479/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
dense_479/Tensordot/concat�
dense_479/Tensordot/stackPack!dense_479/Tensordot/Prod:output:0#dense_479/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
dense_479/Tensordot/stack�
dense_479/Tensordot/transpose	Transposedense_478/Relu:activations:0#dense_479/Tensordot/concat:output:0*
T0*,
_output_shapes
:��������� �2
dense_479/Tensordot/transpose�
dense_479/Tensordot/ReshapeReshape!dense_479/Tensordot/transpose:y:0"dense_479/Tensordot/stack:output:0*
T0*0
_output_shapes
:������������������2
dense_479/Tensordot/Reshape�
dense_479/Tensordot/MatMulMatMul$dense_479/Tensordot/Reshape:output:0*dense_479/Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
dense_479/Tensordot/MatMul�
dense_479/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:�2
dense_479/Tensordot/Const_2�
!dense_479/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2#
!dense_479/Tensordot/concat_1/axis�
dense_479/Tensordot/concat_1ConcatV2%dense_479/Tensordot/GatherV2:output:0$dense_479/Tensordot/Const_2:output:0*dense_479/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
dense_479/Tensordot/concat_1�
dense_479/TensordotReshape$dense_479/Tensordot/MatMul:product:0%dense_479/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:��������� �2
dense_479/Tensordot�
 dense_479/BiasAdd/ReadVariableOpReadVariableOp)dense_479_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02"
 dense_479/BiasAdd/ReadVariableOp�
dense_479/BiasAddAdddense_479/Tensordot:output:0(dense_479/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:��������� �2
dense_479/BiasAdd
tf_op_layer_Pow_59/Pow_59/xConst*
_output_shapes
: *
dtype0*
valueB
 *   A2
tf_op_layer_Pow_59/Pow_59/x�
tf_op_layer_Pow_59/Pow_59Pow$tf_op_layer_Pow_59/Pow_59/x:output:0tf_op_layer_Mul_354/Mul_354:z:0*
T0*
_cloned(*,
_output_shapes
:��������� �2
tf_op_layer_Pow_59/Pow_59�
tf_op_layer_Mul_355/Mul_355Muldense_479/BiasAdd:z:0tf_op_layer_Pow_59/Pow_59:z:0*
T0*
_cloned(*,
_output_shapes
:��������� �2
tf_op_layer_Mul_355/Mul_355�
tf_op_layer_Relu_55/Relu_55Relutf_op_layer_Mul_355/Mul_355:z:0*
T0*
_cloned(*,
_output_shapes
:��������� �2
tf_op_layer_Relu_55/Relu_55�
+tf_op_layer_Max_63/Max_63/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
���������2-
+tf_op_layer_Max_63/Max_63/reduction_indices�
tf_op_layer_Max_63/Max_63Maxinputs_24tf_op_layer_Max_63/Max_63/reduction_indices:output:0*
T0*
_cloned(*+
_output_shapes
:��������� *
	keep_dims(2
tf_op_layer_Max_63/Max_63�
tf_op_layer_Mul_356/Mul_356Mul)tf_op_layer_Relu_55/Relu_55:activations:0"tf_op_layer_Max_63/Max_63:output:0*
T0*
_cloned(*,
_output_shapes
:��������� �2
tf_op_layer_Mul_356/Mul_356x
IdentityIdentitytf_op_layer_Mul_356/Mul_356:z:0*
T0*,
_output_shapes
:��������� �2

Identity"
identityIdentity:output:0*y
_input_shapesh
f:���������:��������� :��������� �::::::::::Q M
'
_output_shapes
:���������
"
_user_specified_name
inputs/0:UQ
+
_output_shapes
:��������� 
"
_user_specified_name
inputs/1:VR
,
_output_shapes
:��������� �
"
_user_specified_name
inputs/2:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: 
�
u
Y__inference_tf_op_layer_strided_slice_481_layer_call_and_return_conditional_losses_461655

inputs
identity�
strided_slice_481/beginConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_481/begin
strided_slice_481/endConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_481/end�
strided_slice_481/stridesConst*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_481/strides�
strided_slice_481StridedSliceinputs strided_slice_481/begin:output:0strided_slice_481/end:output:0"strided_slice_481/strides:output:0*
Index0*
T0*
_cloned(*+
_output_shapes
:��������� *
ellipsis_mask2
strided_slice_481r
IdentityIdentitystrided_slice_481:output:0*
T0*+
_output_shapes
:��������� 2

Identity"
identityIdentity:output:0**
_input_shapes
:��������� :S O
+
_output_shapes
:��������� 
 
_user_specified_nameinputs
�
j
N__inference_tf_op_layer_Pow_59_layer_call_and_return_conditional_losses_461885

inputs
identityY
Pow_59/xConst*
_output_shapes
: *
dtype0*
valueB
 *   A2

Pow_59/xx
Pow_59PowPow_59/x:output:0inputs*
T0*
_cloned(*,
_output_shapes
:��������� �2
Pow_59c
IdentityIdentity
Pow_59:z:0*
T0*,
_output_shapes
:��������� �2

Identity"
identityIdentity:output:0*+
_input_shapes
:��������� �:T P
,
_output_shapes
:��������� �
 
_user_specified_nameinputs
�
u
Y__inference_tf_op_layer_strided_slice_480_layer_call_and_return_conditional_losses_461537

inputs
identity�
strided_slice_480/beginConst*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice_480/begin
strided_slice_480/endConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_480/end�
strided_slice_480/stridesConst*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_480/strides�
strided_slice_480StridedSliceinputs strided_slice_480/begin:output:0strided_slice_480/end:output:0"strided_slice_480/strides:output:0*
Index0*
T0*
_cloned(*+
_output_shapes
:��������� *
ellipsis_mask2
strided_slice_480r
IdentityIdentitystrided_slice_480:output:0*
T0*+
_output_shapes
:��������� 2

Identity"
identityIdentity:output:0**
_input_shapes
:��������� :S O
+
_output_shapes
:��������� 
 
_user_specified_nameinputs
�
�
*__inference_model_119_layer_call_fn_462521
inputs_0
inputs_1
inputs_2
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputs_0inputs_1inputs_2unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7*
Tin
2*
Tout
2*,
_output_shapes
:��������� �*+
_read_only_resource_inputs
		
*-
config_proto

GPU

CPU2*0J 8*N
fIRG
E__inference_model_119_layer_call_and_return_conditional_losses_4620422
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*,
_output_shapes
:��������� �2

Identity"
identityIdentity:output:0*y
_input_shapesh
f:���������:��������� :��������� �:::::::::22
StatefulPartitionedCallStatefulPartitionedCall:Q M
'
_output_shapes
:���������
"
_user_specified_name
inputs/0:UQ
+
_output_shapes
:��������� 
"
_user_specified_name
inputs/1:VR
,
_output_shapes
:��������� �
"
_user_specified_name
inputs/2:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: 
�
Z
>__inference_tf_op_layer_strided_slice_483_layer_call_fn_462572

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*+
_output_shapes
:��������� * 
_read_only_resource_inputs
 *-
config_proto

GPU

CPU2*0J 8*b
f]R[
Y__inference_tf_op_layer_strided_slice_483_layer_call_and_return_conditional_losses_4615532
PartitionedCallp
IdentityIdentityPartitionedCall:output:0*
T0*+
_output_shapes
:��������� 2

Identity"
identityIdentity:output:0**
_input_shapes
:��������� :S O
+
_output_shapes
:��������� 
 
_user_specified_nameinputs
�
h
L__inference_repeat_vector_59_layer_call_and_return_conditional_losses_461516

inputs
identityb
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2
ExpandDims/dim�

ExpandDims
ExpandDimsinputsExpandDims/dim:output:0*
T0*4
_output_shapes"
 :������������������2

ExpandDimsc
stackConst*
_output_shapes
:*
dtype0*!
valueB"          2
stackx
TileTileExpandDims:output:0stack:output:0*
T0*4
_output_shapes"
 :��������� ���������2
Tilen
IdentityIdentityTile:output:0*
T0*4
_output_shapes"
 :��������� ���������2

Identity"
identityIdentity:output:0*/
_input_shapes
:������������������:X T
0
_output_shapes
:������������������
 
_user_specified_nameinputs
�
M
1__inference_repeat_vector_59_layer_call_fn_461522

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*4
_output_shapes"
 :��������� ���������* 
_read_only_resource_inputs
 *-
config_proto

GPU

CPU2*0J 8*U
fPRN
L__inference_repeat_vector_59_layer_call_and_return_conditional_losses_4615162
PartitionedCally
IdentityIdentityPartitionedCall:output:0*
T0*4
_output_shapes"
 :��������� ���������2

Identity"
identityIdentity:output:0*/
_input_shapes
:������������������:X T
0
_output_shapes
:������������������
 
_user_specified_nameinputs
�K
�
E__inference_model_119_layer_call_and_return_conditional_losses_461994
latent_params
conditional_params
	input_240
color_law_461962
dense_476_461966
dense_476_461968
dense_477_461972
dense_477_461974
dense_478_461978
dense_478_461980
dense_479_461983
dense_479_461985
identity��!color_law/StatefulPartitionedCall�!dense_476/StatefulPartitionedCall�!dense_477/StatefulPartitionedCall�!dense_478/StatefulPartitionedCall�!dense_479/StatefulPartitionedCall�
 repeat_vector_59/PartitionedCallPartitionedCalllatent_params*
Tin
2*
Tout
2*+
_output_shapes
:��������� * 
_read_only_resource_inputs
 *-
config_proto

GPU

CPU2*0J 8*U
fPRN
L__inference_repeat_vector_59_layer_call_and_return_conditional_losses_4615162"
 repeat_vector_59/PartitionedCall�
-tf_op_layer_strided_slice_480/PartitionedCallPartitionedCall)repeat_vector_59/PartitionedCall:output:0*
Tin
2*
Tout
2*+
_output_shapes
:��������� * 
_read_only_resource_inputs
 *-
config_proto

GPU

CPU2*0J 8*b
f]R[
Y__inference_tf_op_layer_strided_slice_480_layer_call_and_return_conditional_losses_4615372/
-tf_op_layer_strided_slice_480/PartitionedCall�
-tf_op_layer_strided_slice_483/PartitionedCallPartitionedCall)repeat_vector_59/PartitionedCall:output:0*
Tin
2*
Tout
2*+
_output_shapes
:��������� * 
_read_only_resource_inputs
 *-
config_proto

GPU

CPU2*0J 8*b
f]R[
Y__inference_tf_op_layer_strided_slice_483_layer_call_and_return_conditional_losses_4615532/
-tf_op_layer_strided_slice_483/PartitionedCall�
%tf_op_layer_AddV2_118/PartitionedCallPartitionedCallconditional_params6tf_op_layer_strided_slice_480/PartitionedCall:output:0*
Tin
2*
Tout
2*+
_output_shapes
:��������� * 
_read_only_resource_inputs
 *-
config_proto

GPU

CPU2*0J 8*Z
fURS
Q__inference_tf_op_layer_AddV2_118_layer_call_and_return_conditional_losses_4615672'
%tf_op_layer_AddV2_118/PartitionedCall�
-tf_op_layer_strided_slice_482/PartitionedCallPartitionedCall)repeat_vector_59/PartitionedCall:output:0*
Tin
2*
Tout
2*+
_output_shapes
:��������� * 
_read_only_resource_inputs
 *-
config_proto

GPU

CPU2*0J 8*b
f]R[
Y__inference_tf_op_layer_strided_slice_482_layer_call_and_return_conditional_losses_4615842/
-tf_op_layer_strided_slice_482/PartitionedCall�
concatenate_179/PartitionedCallPartitionedCall6tf_op_layer_strided_slice_483/PartitionedCall:output:0.tf_op_layer_AddV2_118/PartitionedCall:output:0*
Tin
2*
Tout
2*+
_output_shapes
:��������� * 
_read_only_resource_inputs
 *-
config_proto

GPU

CPU2*0J 8*T
fORM
K__inference_concatenate_179_layer_call_and_return_conditional_losses_4615992!
concatenate_179/PartitionedCall�
!color_law/StatefulPartitionedCallStatefulPartitionedCall6tf_op_layer_strided_slice_482/PartitionedCall:output:0color_law_461962*
Tin
2*
Tout
2*,
_output_shapes
:��������� �*#
_read_only_resource_inputs
*-
config_proto

GPU

CPU2*0J 8*N
fIRG
E__inference_color_law_layer_call_and_return_conditional_losses_4616352#
!color_law/StatefulPartitionedCall�
-tf_op_layer_strided_slice_481/PartitionedCallPartitionedCall)repeat_vector_59/PartitionedCall:output:0*
Tin
2*
Tout
2*+
_output_shapes
:��������� * 
_read_only_resource_inputs
 *-
config_proto

GPU

CPU2*0J 8*b
f]R[
Y__inference_tf_op_layer_strided_slice_481_layer_call_and_return_conditional_losses_4616552/
-tf_op_layer_strided_slice_481/PartitionedCall�
!dense_476/StatefulPartitionedCallStatefulPartitionedCall(concatenate_179/PartitionedCall:output:0dense_476_461966dense_476_461968*
Tin
2*
Tout
2*+
_output_shapes
:���������  *$
_read_only_resource_inputs
*-
config_proto

GPU

CPU2*0J 8*N
fIRG
E__inference_dense_476_layer_call_and_return_conditional_losses_4616942#
!dense_476/StatefulPartitionedCall�
%tf_op_layer_AddV2_119/PartitionedCallPartitionedCall*color_law/StatefulPartitionedCall:output:06tf_op_layer_strided_slice_481/PartitionedCall:output:0*
Tin
2*
Tout
2*,
_output_shapes
:��������� �* 
_read_only_resource_inputs
 *-
config_proto

GPU

CPU2*0J 8*Z
fURS
Q__inference_tf_op_layer_AddV2_119_layer_call_and_return_conditional_losses_4617162'
%tf_op_layer_AddV2_119/PartitionedCall�
!dense_477/StatefulPartitionedCallStatefulPartitionedCall*dense_476/StatefulPartitionedCall:output:0dense_477_461972dense_477_461974*
Tin
2*
Tout
2*,
_output_shapes
:��������� �*$
_read_only_resource_inputs
*-
config_proto

GPU

CPU2*0J 8*N
fIRG
E__inference_dense_477_layer_call_and_return_conditional_losses_4617562#
!dense_477/StatefulPartitionedCall�
#tf_op_layer_Mul_354/PartitionedCallPartitionedCall.tf_op_layer_AddV2_119/PartitionedCall:output:0*
Tin
2*
Tout
2*,
_output_shapes
:��������� �* 
_read_only_resource_inputs
 *-
config_proto

GPU

CPU2*0J 8*X
fSRQ
O__inference_tf_op_layer_Mul_354_layer_call_and_return_conditional_losses_4617782%
#tf_op_layer_Mul_354/PartitionedCall�
!dense_478/StatefulPartitionedCallStatefulPartitionedCall*dense_477/StatefulPartitionedCall:output:0dense_478_461978dense_478_461980*
Tin
2*
Tout
2*,
_output_shapes
:��������� �*$
_read_only_resource_inputs
*-
config_proto

GPU

CPU2*0J 8*N
fIRG
E__inference_dense_478_layer_call_and_return_conditional_losses_4618172#
!dense_478/StatefulPartitionedCall�
!dense_479/StatefulPartitionedCallStatefulPartitionedCall*dense_478/StatefulPartitionedCall:output:0dense_479_461983dense_479_461985*
Tin
2*
Tout
2*,
_output_shapes
:��������� �*$
_read_only_resource_inputs
*-
config_proto

GPU

CPU2*0J 8*N
fIRG
E__inference_dense_479_layer_call_and_return_conditional_losses_4618632#
!dense_479/StatefulPartitionedCall�
"tf_op_layer_Pow_59/PartitionedCallPartitionedCall,tf_op_layer_Mul_354/PartitionedCall:output:0*
Tin
2*
Tout
2*,
_output_shapes
:��������� �* 
_read_only_resource_inputs
 *-
config_proto

GPU

CPU2*0J 8*W
fRRP
N__inference_tf_op_layer_Pow_59_layer_call_and_return_conditional_losses_4618852$
"tf_op_layer_Pow_59/PartitionedCall�
#tf_op_layer_Mul_355/PartitionedCallPartitionedCall*dense_479/StatefulPartitionedCall:output:0+tf_op_layer_Pow_59/PartitionedCall:output:0*
Tin
2*
Tout
2*,
_output_shapes
:��������� �* 
_read_only_resource_inputs
 *-
config_proto

GPU

CPU2*0J 8*X
fSRQ
O__inference_tf_op_layer_Mul_355_layer_call_and_return_conditional_losses_4618992%
#tf_op_layer_Mul_355/PartitionedCall�
#tf_op_layer_Relu_55/PartitionedCallPartitionedCall,tf_op_layer_Mul_355/PartitionedCall:output:0*
Tin
2*
Tout
2*,
_output_shapes
:��������� �* 
_read_only_resource_inputs
 *-
config_proto

GPU

CPU2*0J 8*X
fSRQ
O__inference_tf_op_layer_Relu_55_layer_call_and_return_conditional_losses_4619132%
#tf_op_layer_Relu_55/PartitionedCall�
"tf_op_layer_Max_63/PartitionedCallPartitionedCall	input_240*
Tin
2*
Tout
2*+
_output_shapes
:��������� * 
_read_only_resource_inputs
 *-
config_proto

GPU

CPU2*0J 8*W
fRRP
N__inference_tf_op_layer_Max_63_layer_call_and_return_conditional_losses_4619272$
"tf_op_layer_Max_63/PartitionedCall�
#tf_op_layer_Mul_356/PartitionedCallPartitionedCall,tf_op_layer_Relu_55/PartitionedCall:output:0+tf_op_layer_Max_63/PartitionedCall:output:0*
Tin
2*
Tout
2*,
_output_shapes
:��������� �* 
_read_only_resource_inputs
 *-
config_proto

GPU

CPU2*0J 8*X
fSRQ
O__inference_tf_op_layer_Mul_356_layer_call_and_return_conditional_losses_4619412%
#tf_op_layer_Mul_356/PartitionedCall�
IdentityIdentity,tf_op_layer_Mul_356/PartitionedCall:output:0"^color_law/StatefulPartitionedCall"^dense_476/StatefulPartitionedCall"^dense_477/StatefulPartitionedCall"^dense_478/StatefulPartitionedCall"^dense_479/StatefulPartitionedCall*
T0*,
_output_shapes
:��������� �2

Identity"
identityIdentity:output:0*y
_input_shapesh
f:���������:��������� :��������� �:::::::::2F
!color_law/StatefulPartitionedCall!color_law/StatefulPartitionedCall2F
!dense_476/StatefulPartitionedCall!dense_476/StatefulPartitionedCall2F
!dense_477/StatefulPartitionedCall!dense_477/StatefulPartitionedCall2F
!dense_478/StatefulPartitionedCall!dense_478/StatefulPartitionedCall2F
!dense_479/StatefulPartitionedCall!dense_479/StatefulPartitionedCall:V R
'
_output_shapes
:���������
'
_user_specified_namelatent_params:_[
+
_output_shapes
:��������� 
,
_user_specified_nameconditional_params:WS
,
_output_shapes
:��������� �
#
_user_specified_name	input_240:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: 
�(
�
__inference__traced_save_462951
file_prefix/
+savev2_dense_476_kernel_read_readvariableop-
)savev2_dense_476_bias_read_readvariableop2
.savev2_color_law_62_kernel_read_readvariableop/
+savev2_dense_477_kernel_read_readvariableop-
)savev2_dense_477_bias_read_readvariableop/
+savev2_dense_478_kernel_read_readvariableop-
)savev2_dense_478_bias_read_readvariableop/
+savev2_dense_479_kernel_read_readvariableop-
)savev2_dense_479_bias_read_readvariableop
savev2_1_const

identity_1��MergeV2Checkpoints�SaveV2�SaveV2_1�
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*2
StaticRegexFullMatchc
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part2
Const�
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*<
value3B1 B+_temp_3facea277306445badefa689c1ae3d86/part2	
Const_1�
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: 2
Selectt

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shard�
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename�
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:	*
dtype0*�
value�B�	B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE2
SaveV2/tensor_names�
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:	*
dtype0*%
valueB	B B B B B B B B B 2
SaveV2/shape_and_slices�
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0+savev2_dense_476_kernel_read_readvariableop)savev2_dense_476_bias_read_readvariableop.savev2_color_law_62_kernel_read_readvariableop+savev2_dense_477_kernel_read_readvariableop)savev2_dense_477_bias_read_readvariableop+savev2_dense_478_kernel_read_readvariableop)savev2_dense_478_bias_read_readvariableop+savev2_dense_479_kernel_read_readvariableop)savev2_dense_479_bias_read_readvariableop"/device:CPU:0*
_output_shapes
 *
dtypes
2	2
SaveV2�
ShardedFilename_1/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B :2
ShardedFilename_1/shard�
ShardedFilename_1ShardedFilenameStringJoin:output:0 ShardedFilename_1/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename_1�
SaveV2_1/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*1
value(B&B_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2_1/tensor_names�
SaveV2_1/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueB
B 2
SaveV2_1/shape_and_slices�
SaveV2_1SaveV2ShardedFilename_1:filename:0SaveV2_1/tensor_names:output:0"SaveV2_1/shape_and_slices:output:0savev2_1_const^SaveV2"/device:CPU:0*
_output_shapes
 *
dtypes
22

SaveV2_1�
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0ShardedFilename_1:filename:0^SaveV2	^SaveV2_1"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes�
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix	^SaveV2_1"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identity�

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints^SaveV2	^SaveV2_1*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*j
_input_shapesY
W: : : :	�:	 �:�:
��:�:
��:�: 2(
MergeV2CheckpointsMergeV2Checkpoints2
SaveV2SaveV22
SaveV2_1SaveV2_1:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:$ 

_output_shapes

: : 

_output_shapes
: :%!

_output_shapes
:	�:%!

_output_shapes
:	 �:!

_output_shapes	
:�:&"
 
_output_shapes
:
��:!

_output_shapes	
:�:&"
 
_output_shapes
:
��:!	

_output_shapes	
:�:


_output_shapes
: 
�
{
Q__inference_tf_op_layer_AddV2_119_layer_call_and_return_conditional_losses_461716

inputs
inputs_1
identityw
	AddV2_119AddV2inputsinputs_1*
T0*
_cloned(*,
_output_shapes
:��������� �2
	AddV2_119f
IdentityIdentityAddV2_119:z:0*
T0*,
_output_shapes
:��������� �2

Identity"
identityIdentity:output:0*B
_input_shapes1
/:��������� �:��������� :T P
,
_output_shapes
:��������� �
 
_user_specified_nameinputs:SO
+
_output_shapes
:��������� 
 
_user_specified_nameinputs
�

*__inference_dense_476_layer_call_fn_462650

inputs
unknown
	unknown_0
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*+
_output_shapes
:���������  *$
_read_only_resource_inputs
*-
config_proto

GPU

CPU2*0J 8*N
fIRG
E__inference_dense_476_layer_call_and_return_conditional_losses_4616942
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:���������  2

Identity"
identityIdentity:output:0*2
_input_shapes!
:��������� ::22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:��������� 
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
�
u
Y__inference_tf_op_layer_strided_slice_480_layer_call_and_return_conditional_losses_462554

inputs
identity�
strided_slice_480/beginConst*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice_480/begin
strided_slice_480/endConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_480/end�
strided_slice_480/stridesConst*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_480/strides�
strided_slice_480StridedSliceinputs strided_slice_480/begin:output:0strided_slice_480/end:output:0"strided_slice_480/strides:output:0*
Index0*
T0*
_cloned(*+
_output_shapes
:��������� *
ellipsis_mask2
strided_slice_480r
IdentityIdentitystrided_slice_480:output:0*
T0*+
_output_shapes
:��������� 2

Identity"
identityIdentity:output:0**
_input_shapes
:��������� :S O
+
_output_shapes
:��������� 
 
_user_specified_nameinputs
�
b
6__inference_tf_op_layer_AddV2_119_layer_call_fn_462749
inputs_0
inputs_1
identity�
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*,
_output_shapes
:��������� �* 
_read_only_resource_inputs
 *-
config_proto

GPU

CPU2*0J 8*Z
fURS
Q__inference_tf_op_layer_AddV2_119_layer_call_and_return_conditional_losses_4617162
PartitionedCallq
IdentityIdentityPartitionedCall:output:0*
T0*,
_output_shapes
:��������� �2

Identity"
identityIdentity:output:0*B
_input_shapes1
/:��������� �:��������� :V R
,
_output_shapes
:��������� �
"
_user_specified_name
inputs/0:UQ
+
_output_shapes
:��������� 
"
_user_specified_name
inputs/1
�K
�
E__inference_model_119_layer_call_and_return_conditional_losses_462110

inputs
inputs_1
inputs_2
color_law_462078
dense_476_462082
dense_476_462084
dense_477_462088
dense_477_462090
dense_478_462094
dense_478_462096
dense_479_462099
dense_479_462101
identity��!color_law/StatefulPartitionedCall�!dense_476/StatefulPartitionedCall�!dense_477/StatefulPartitionedCall�!dense_478/StatefulPartitionedCall�!dense_479/StatefulPartitionedCall�
 repeat_vector_59/PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*+
_output_shapes
:��������� * 
_read_only_resource_inputs
 *-
config_proto

GPU

CPU2*0J 8*U
fPRN
L__inference_repeat_vector_59_layer_call_and_return_conditional_losses_4615162"
 repeat_vector_59/PartitionedCall�
-tf_op_layer_strided_slice_480/PartitionedCallPartitionedCall)repeat_vector_59/PartitionedCall:output:0*
Tin
2*
Tout
2*+
_output_shapes
:��������� * 
_read_only_resource_inputs
 *-
config_proto

GPU

CPU2*0J 8*b
f]R[
Y__inference_tf_op_layer_strided_slice_480_layer_call_and_return_conditional_losses_4615372/
-tf_op_layer_strided_slice_480/PartitionedCall�
-tf_op_layer_strided_slice_483/PartitionedCallPartitionedCall)repeat_vector_59/PartitionedCall:output:0*
Tin
2*
Tout
2*+
_output_shapes
:��������� * 
_read_only_resource_inputs
 *-
config_proto

GPU

CPU2*0J 8*b
f]R[
Y__inference_tf_op_layer_strided_slice_483_layer_call_and_return_conditional_losses_4615532/
-tf_op_layer_strided_slice_483/PartitionedCall�
%tf_op_layer_AddV2_118/PartitionedCallPartitionedCallinputs_16tf_op_layer_strided_slice_480/PartitionedCall:output:0*
Tin
2*
Tout
2*+
_output_shapes
:��������� * 
_read_only_resource_inputs
 *-
config_proto

GPU

CPU2*0J 8*Z
fURS
Q__inference_tf_op_layer_AddV2_118_layer_call_and_return_conditional_losses_4615672'
%tf_op_layer_AddV2_118/PartitionedCall�
-tf_op_layer_strided_slice_482/PartitionedCallPartitionedCall)repeat_vector_59/PartitionedCall:output:0*
Tin
2*
Tout
2*+
_output_shapes
:��������� * 
_read_only_resource_inputs
 *-
config_proto

GPU

CPU2*0J 8*b
f]R[
Y__inference_tf_op_layer_strided_slice_482_layer_call_and_return_conditional_losses_4615842/
-tf_op_layer_strided_slice_482/PartitionedCall�
concatenate_179/PartitionedCallPartitionedCall6tf_op_layer_strided_slice_483/PartitionedCall:output:0.tf_op_layer_AddV2_118/PartitionedCall:output:0*
Tin
2*
Tout
2*+
_output_shapes
:��������� * 
_read_only_resource_inputs
 *-
config_proto

GPU

CPU2*0J 8*T
fORM
K__inference_concatenate_179_layer_call_and_return_conditional_losses_4615992!
concatenate_179/PartitionedCall�
!color_law/StatefulPartitionedCallStatefulPartitionedCall6tf_op_layer_strided_slice_482/PartitionedCall:output:0color_law_462078*
Tin
2*
Tout
2*,
_output_shapes
:��������� �*#
_read_only_resource_inputs
*-
config_proto

GPU

CPU2*0J 8*N
fIRG
E__inference_color_law_layer_call_and_return_conditional_losses_4616352#
!color_law/StatefulPartitionedCall�
-tf_op_layer_strided_slice_481/PartitionedCallPartitionedCall)repeat_vector_59/PartitionedCall:output:0*
Tin
2*
Tout
2*+
_output_shapes
:��������� * 
_read_only_resource_inputs
 *-
config_proto

GPU

CPU2*0J 8*b
f]R[
Y__inference_tf_op_layer_strided_slice_481_layer_call_and_return_conditional_losses_4616552/
-tf_op_layer_strided_slice_481/PartitionedCall�
!dense_476/StatefulPartitionedCallStatefulPartitionedCall(concatenate_179/PartitionedCall:output:0dense_476_462082dense_476_462084*
Tin
2*
Tout
2*+
_output_shapes
:���������  *$
_read_only_resource_inputs
*-
config_proto

GPU

CPU2*0J 8*N
fIRG
E__inference_dense_476_layer_call_and_return_conditional_losses_4616942#
!dense_476/StatefulPartitionedCall�
%tf_op_layer_AddV2_119/PartitionedCallPartitionedCall*color_law/StatefulPartitionedCall:output:06tf_op_layer_strided_slice_481/PartitionedCall:output:0*
Tin
2*
Tout
2*,
_output_shapes
:��������� �* 
_read_only_resource_inputs
 *-
config_proto

GPU

CPU2*0J 8*Z
fURS
Q__inference_tf_op_layer_AddV2_119_layer_call_and_return_conditional_losses_4617162'
%tf_op_layer_AddV2_119/PartitionedCall�
!dense_477/StatefulPartitionedCallStatefulPartitionedCall*dense_476/StatefulPartitionedCall:output:0dense_477_462088dense_477_462090*
Tin
2*
Tout
2*,
_output_shapes
:��������� �*$
_read_only_resource_inputs
*-
config_proto

GPU

CPU2*0J 8*N
fIRG
E__inference_dense_477_layer_call_and_return_conditional_losses_4617562#
!dense_477/StatefulPartitionedCall�
#tf_op_layer_Mul_354/PartitionedCallPartitionedCall.tf_op_layer_AddV2_119/PartitionedCall:output:0*
Tin
2*
Tout
2*,
_output_shapes
:��������� �* 
_read_only_resource_inputs
 *-
config_proto

GPU

CPU2*0J 8*X
fSRQ
O__inference_tf_op_layer_Mul_354_layer_call_and_return_conditional_losses_4617782%
#tf_op_layer_Mul_354/PartitionedCall�
!dense_478/StatefulPartitionedCallStatefulPartitionedCall*dense_477/StatefulPartitionedCall:output:0dense_478_462094dense_478_462096*
Tin
2*
Tout
2*,
_output_shapes
:��������� �*$
_read_only_resource_inputs
*-
config_proto

GPU

CPU2*0J 8*N
fIRG
E__inference_dense_478_layer_call_and_return_conditional_losses_4618172#
!dense_478/StatefulPartitionedCall�
!dense_479/StatefulPartitionedCallStatefulPartitionedCall*dense_478/StatefulPartitionedCall:output:0dense_479_462099dense_479_462101*
Tin
2*
Tout
2*,
_output_shapes
:��������� �*$
_read_only_resource_inputs
*-
config_proto

GPU

CPU2*0J 8*N
fIRG
E__inference_dense_479_layer_call_and_return_conditional_losses_4618632#
!dense_479/StatefulPartitionedCall�
"tf_op_layer_Pow_59/PartitionedCallPartitionedCall,tf_op_layer_Mul_354/PartitionedCall:output:0*
Tin
2*
Tout
2*,
_output_shapes
:��������� �* 
_read_only_resource_inputs
 *-
config_proto

GPU

CPU2*0J 8*W
fRRP
N__inference_tf_op_layer_Pow_59_layer_call_and_return_conditional_losses_4618852$
"tf_op_layer_Pow_59/PartitionedCall�
#tf_op_layer_Mul_355/PartitionedCallPartitionedCall*dense_479/StatefulPartitionedCall:output:0+tf_op_layer_Pow_59/PartitionedCall:output:0*
Tin
2*
Tout
2*,
_output_shapes
:��������� �* 
_read_only_resource_inputs
 *-
config_proto

GPU

CPU2*0J 8*X
fSRQ
O__inference_tf_op_layer_Mul_355_layer_call_and_return_conditional_losses_4618992%
#tf_op_layer_Mul_355/PartitionedCall�
#tf_op_layer_Relu_55/PartitionedCallPartitionedCall,tf_op_layer_Mul_355/PartitionedCall:output:0*
Tin
2*
Tout
2*,
_output_shapes
:��������� �* 
_read_only_resource_inputs
 *-
config_proto

GPU

CPU2*0J 8*X
fSRQ
O__inference_tf_op_layer_Relu_55_layer_call_and_return_conditional_losses_4619132%
#tf_op_layer_Relu_55/PartitionedCall�
"tf_op_layer_Max_63/PartitionedCallPartitionedCallinputs_2*
Tin
2*
Tout
2*+
_output_shapes
:��������� * 
_read_only_resource_inputs
 *-
config_proto

GPU

CPU2*0J 8*W
fRRP
N__inference_tf_op_layer_Max_63_layer_call_and_return_conditional_losses_4619272$
"tf_op_layer_Max_63/PartitionedCall�
#tf_op_layer_Mul_356/PartitionedCallPartitionedCall,tf_op_layer_Relu_55/PartitionedCall:output:0+tf_op_layer_Max_63/PartitionedCall:output:0*
Tin
2*
Tout
2*,
_output_shapes
:��������� �* 
_read_only_resource_inputs
 *-
config_proto

GPU

CPU2*0J 8*X
fSRQ
O__inference_tf_op_layer_Mul_356_layer_call_and_return_conditional_losses_4619412%
#tf_op_layer_Mul_356/PartitionedCall�
IdentityIdentity,tf_op_layer_Mul_356/PartitionedCall:output:0"^color_law/StatefulPartitionedCall"^dense_476/StatefulPartitionedCall"^dense_477/StatefulPartitionedCall"^dense_478/StatefulPartitionedCall"^dense_479/StatefulPartitionedCall*
T0*,
_output_shapes
:��������� �2

Identity"
identityIdentity:output:0*y
_input_shapesh
f:���������:��������� :��������� �:::::::::2F
!color_law/StatefulPartitionedCall!color_law/StatefulPartitionedCall2F
!dense_476/StatefulPartitionedCall!dense_476/StatefulPartitionedCall2F
!dense_477/StatefulPartitionedCall!dense_477/StatefulPartitionedCall2F
!dense_478/StatefulPartitionedCall!dense_478/StatefulPartitionedCall2F
!dense_479/StatefulPartitionedCall!dense_479/StatefulPartitionedCall:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs:SO
+
_output_shapes
:��������� 
 
_user_specified_nameinputs:TP
,
_output_shapes
:��������� �
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: 
�
k
O__inference_tf_op_layer_Mul_354_layer_call_and_return_conditional_losses_462795

inputs
identity[
	Mul_354/xConst*
_output_shapes
: *
dtype0*
valueB
 *��̾2
	Mul_354/x{
Mul_354MulMul_354/x:output:0inputs*
T0*
_cloned(*,
_output_shapes
:��������� �2	
Mul_354d
IdentityIdentityMul_354:z:0*
T0*,
_output_shapes
:��������� �2

Identity"
identityIdentity:output:0*+
_input_shapes
:��������� �:T P
,
_output_shapes
:��������� �
 
_user_specified_nameinputs
�
�
E__inference_color_law_layer_call_and_return_conditional_losses_462677

inputs%
!tensordot_readvariableop_resource
identity��
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes
:	�*
dtype02
Tensordot/ReadVariableOpj
Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
Tensordot/axesq
Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
Tensordot/freeX
Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:2
Tensordot/Shapet
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/GatherV2/axis�
Tensordot/GatherV2GatherV2Tensordot/Shape:output:0Tensordot/free:output:0 Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
Tensordot/GatherV2x
Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/GatherV2_1/axis�
Tensordot/GatherV2_1GatherV2Tensordot/Shape:output:0Tensordot/axes:output:0"Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
Tensordot/GatherV2_1l
Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot/Const�
Tensordot/ProdProdTensordot/GatherV2:output:0Tensordot/Const:output:0*
T0*
_output_shapes
: 2
Tensordot/Prodp
Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot/Const_1�
Tensordot/Prod_1ProdTensordot/GatherV2_1:output:0Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
Tensordot/Prod_1p
Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/concat/axis�
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat�
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
Tensordot/stack�
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*+
_output_shapes
:��������� 2
Tensordot/transpose�
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:������������������2
Tensordot/Reshape�
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
Tensordot/MatMulq
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:�2
Tensordot/Const_2t
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/concat_1/axis�
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat_1�
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*,
_output_shapes
:��������� �2
	Tensordotk
IdentityIdentityTensordot:output:0*
T0*,
_output_shapes
:��������� �2

Identity"
identityIdentity:output:0*.
_input_shapes
:��������� ::S O
+
_output_shapes
:��������� 
 
_user_specified_nameinputs:

_output_shapes
: 
�
Z
>__inference_tf_op_layer_strided_slice_481_layer_call_fn_462697

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*+
_output_shapes
:��������� * 
_read_only_resource_inputs
 *-
config_proto

GPU

CPU2*0J 8*b
f]R[
Y__inference_tf_op_layer_strided_slice_481_layer_call_and_return_conditional_losses_4616552
PartitionedCallp
IdentityIdentityPartitionedCall:output:0*
T0*+
_output_shapes
:��������� 2

Identity"
identityIdentity:output:0**
_input_shapes
:��������� :S O
+
_output_shapes
:��������� 
 
_user_specified_nameinputs
�
k
O__inference_tf_op_layer_Relu_55_layer_call_and_return_conditional_losses_462867

inputs
identityh
Relu_55Reluinputs*
T0*
_cloned(*,
_output_shapes
:��������� �2	
Relu_55n
IdentityIdentityRelu_55:activations:0*
T0*,
_output_shapes
:��������� �2

Identity"
identityIdentity:output:0*+
_input_shapes
:��������� �:T P
,
_output_shapes
:��������� �
 
_user_specified_nameinputs
��
�
!__inference__wrapped_model_461507
latent_params
conditional_params
	input_2409
5model_119_color_law_tensordot_readvariableop_resource9
5model_119_dense_476_tensordot_readvariableop_resource7
3model_119_dense_476_biasadd_readvariableop_resource9
5model_119_dense_477_tensordot_readvariableop_resource7
3model_119_dense_477_biasadd_readvariableop_resource9
5model_119_dense_478_tensordot_readvariableop_resource7
3model_119_dense_478_biasadd_readvariableop_resource9
5model_119_dense_479_tensordot_readvariableop_resource7
3model_119_dense_479_biasadd_readvariableop_resource
identity��
)model_119/repeat_vector_59/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2+
)model_119/repeat_vector_59/ExpandDims/dim�
%model_119/repeat_vector_59/ExpandDims
ExpandDimslatent_params2model_119/repeat_vector_59/ExpandDims/dim:output:0*
T0*+
_output_shapes
:���������2'
%model_119/repeat_vector_59/ExpandDims�
 model_119/repeat_vector_59/stackConst*
_output_shapes
:*
dtype0*!
valueB"          2"
 model_119/repeat_vector_59/stack�
model_119/repeat_vector_59/TileTile.model_119/repeat_vector_59/ExpandDims:output:0)model_119/repeat_vector_59/stack:output:0*
T0*+
_output_shapes
:��������� 2!
model_119/repeat_vector_59/Tile�
?model_119/tf_op_layer_strided_slice_480/strided_slice_480/beginConst*
_output_shapes
:*
dtype0*
valueB"        2A
?model_119/tf_op_layer_strided_slice_480/strided_slice_480/begin�
=model_119/tf_op_layer_strided_slice_480/strided_slice_480/endConst*
_output_shapes
:*
dtype0*
valueB"       2?
=model_119/tf_op_layer_strided_slice_480/strided_slice_480/end�
Amodel_119/tf_op_layer_strided_slice_480/strided_slice_480/stridesConst*
_output_shapes
:*
dtype0*
valueB"      2C
Amodel_119/tf_op_layer_strided_slice_480/strided_slice_480/strides�
9model_119/tf_op_layer_strided_slice_480/strided_slice_480StridedSlice(model_119/repeat_vector_59/Tile:output:0Hmodel_119/tf_op_layer_strided_slice_480/strided_slice_480/begin:output:0Fmodel_119/tf_op_layer_strided_slice_480/strided_slice_480/end:output:0Jmodel_119/tf_op_layer_strided_slice_480/strided_slice_480/strides:output:0*
Index0*
T0*
_cloned(*+
_output_shapes
:��������� *
ellipsis_mask2;
9model_119/tf_op_layer_strided_slice_480/strided_slice_480�
?model_119/tf_op_layer_strided_slice_483/strided_slice_483/beginConst*
_output_shapes
:*
dtype0*
valueB"       2A
?model_119/tf_op_layer_strided_slice_483/strided_slice_483/begin�
=model_119/tf_op_layer_strided_slice_483/strided_slice_483/endConst*
_output_shapes
:*
dtype0*
valueB"        2?
=model_119/tf_op_layer_strided_slice_483/strided_slice_483/end�
Amodel_119/tf_op_layer_strided_slice_483/strided_slice_483/stridesConst*
_output_shapes
:*
dtype0*
valueB"      2C
Amodel_119/tf_op_layer_strided_slice_483/strided_slice_483/strides�
9model_119/tf_op_layer_strided_slice_483/strided_slice_483StridedSlice(model_119/repeat_vector_59/Tile:output:0Hmodel_119/tf_op_layer_strided_slice_483/strided_slice_483/begin:output:0Fmodel_119/tf_op_layer_strided_slice_483/strided_slice_483/end:output:0Jmodel_119/tf_op_layer_strided_slice_483/strided_slice_483/strides:output:0*
Index0*
T0*
_cloned(*+
_output_shapes
:��������� *
ellipsis_mask*
end_mask2;
9model_119/tf_op_layer_strided_slice_483/strided_slice_483�
)model_119/tf_op_layer_AddV2_118/AddV2_118AddV2conditional_paramsBmodel_119/tf_op_layer_strided_slice_480/strided_slice_480:output:0*
T0*
_cloned(*+
_output_shapes
:��������� 2+
)model_119/tf_op_layer_AddV2_118/AddV2_118�
?model_119/tf_op_layer_strided_slice_482/strided_slice_482/beginConst*
_output_shapes
:*
dtype0*
valueB"       2A
?model_119/tf_op_layer_strided_slice_482/strided_slice_482/begin�
=model_119/tf_op_layer_strided_slice_482/strided_slice_482/endConst*
_output_shapes
:*
dtype0*
valueB"       2?
=model_119/tf_op_layer_strided_slice_482/strided_slice_482/end�
Amodel_119/tf_op_layer_strided_slice_482/strided_slice_482/stridesConst*
_output_shapes
:*
dtype0*
valueB"      2C
Amodel_119/tf_op_layer_strided_slice_482/strided_slice_482/strides�
9model_119/tf_op_layer_strided_slice_482/strided_slice_482StridedSlice(model_119/repeat_vector_59/Tile:output:0Hmodel_119/tf_op_layer_strided_slice_482/strided_slice_482/begin:output:0Fmodel_119/tf_op_layer_strided_slice_482/strided_slice_482/end:output:0Jmodel_119/tf_op_layer_strided_slice_482/strided_slice_482/strides:output:0*
Index0*
T0*
_cloned(*+
_output_shapes
:��������� *
ellipsis_mask2;
9model_119/tf_op_layer_strided_slice_482/strided_slice_482�
%model_119/concatenate_179/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2'
%model_119/concatenate_179/concat/axis�
 model_119/concatenate_179/concatConcatV2Bmodel_119/tf_op_layer_strided_slice_483/strided_slice_483:output:0-model_119/tf_op_layer_AddV2_118/AddV2_118:z:0.model_119/concatenate_179/concat/axis:output:0*
N*
T0*+
_output_shapes
:��������� 2"
 model_119/concatenate_179/concat�
,model_119/color_law/Tensordot/ReadVariableOpReadVariableOp5model_119_color_law_tensordot_readvariableop_resource*
_output_shapes
:	�*
dtype02.
,model_119/color_law/Tensordot/ReadVariableOp�
"model_119/color_law/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2$
"model_119/color_law/Tensordot/axes�
"model_119/color_law/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2$
"model_119/color_law/Tensordot/free�
#model_119/color_law/Tensordot/ShapeShapeBmodel_119/tf_op_layer_strided_slice_482/strided_slice_482:output:0*
T0*
_output_shapes
:2%
#model_119/color_law/Tensordot/Shape�
+model_119/color_law/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2-
+model_119/color_law/Tensordot/GatherV2/axis�
&model_119/color_law/Tensordot/GatherV2GatherV2,model_119/color_law/Tensordot/Shape:output:0+model_119/color_law/Tensordot/free:output:04model_119/color_law/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2(
&model_119/color_law/Tensordot/GatherV2�
-model_119/color_law/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2/
-model_119/color_law/Tensordot/GatherV2_1/axis�
(model_119/color_law/Tensordot/GatherV2_1GatherV2,model_119/color_law/Tensordot/Shape:output:0+model_119/color_law/Tensordot/axes:output:06model_119/color_law/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2*
(model_119/color_law/Tensordot/GatherV2_1�
#model_119/color_law/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2%
#model_119/color_law/Tensordot/Const�
"model_119/color_law/Tensordot/ProdProd/model_119/color_law/Tensordot/GatherV2:output:0,model_119/color_law/Tensordot/Const:output:0*
T0*
_output_shapes
: 2$
"model_119/color_law/Tensordot/Prod�
%model_119/color_law/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2'
%model_119/color_law/Tensordot/Const_1�
$model_119/color_law/Tensordot/Prod_1Prod1model_119/color_law/Tensordot/GatherV2_1:output:0.model_119/color_law/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2&
$model_119/color_law/Tensordot/Prod_1�
)model_119/color_law/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2+
)model_119/color_law/Tensordot/concat/axis�
$model_119/color_law/Tensordot/concatConcatV2+model_119/color_law/Tensordot/free:output:0+model_119/color_law/Tensordot/axes:output:02model_119/color_law/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2&
$model_119/color_law/Tensordot/concat�
#model_119/color_law/Tensordot/stackPack+model_119/color_law/Tensordot/Prod:output:0-model_119/color_law/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2%
#model_119/color_law/Tensordot/stack�
'model_119/color_law/Tensordot/transpose	TransposeBmodel_119/tf_op_layer_strided_slice_482/strided_slice_482:output:0-model_119/color_law/Tensordot/concat:output:0*
T0*+
_output_shapes
:��������� 2)
'model_119/color_law/Tensordot/transpose�
%model_119/color_law/Tensordot/ReshapeReshape+model_119/color_law/Tensordot/transpose:y:0,model_119/color_law/Tensordot/stack:output:0*
T0*0
_output_shapes
:������������������2'
%model_119/color_law/Tensordot/Reshape�
$model_119/color_law/Tensordot/MatMulMatMul.model_119/color_law/Tensordot/Reshape:output:04model_119/color_law/Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2&
$model_119/color_law/Tensordot/MatMul�
%model_119/color_law/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:�2'
%model_119/color_law/Tensordot/Const_2�
+model_119/color_law/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2-
+model_119/color_law/Tensordot/concat_1/axis�
&model_119/color_law/Tensordot/concat_1ConcatV2/model_119/color_law/Tensordot/GatherV2:output:0.model_119/color_law/Tensordot/Const_2:output:04model_119/color_law/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2(
&model_119/color_law/Tensordot/concat_1�
model_119/color_law/TensordotReshape.model_119/color_law/Tensordot/MatMul:product:0/model_119/color_law/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:��������� �2
model_119/color_law/Tensordot�
?model_119/tf_op_layer_strided_slice_481/strided_slice_481/beginConst*
_output_shapes
:*
dtype0*
valueB"       2A
?model_119/tf_op_layer_strided_slice_481/strided_slice_481/begin�
=model_119/tf_op_layer_strided_slice_481/strided_slice_481/endConst*
_output_shapes
:*
dtype0*
valueB"       2?
=model_119/tf_op_layer_strided_slice_481/strided_slice_481/end�
Amodel_119/tf_op_layer_strided_slice_481/strided_slice_481/stridesConst*
_output_shapes
:*
dtype0*
valueB"      2C
Amodel_119/tf_op_layer_strided_slice_481/strided_slice_481/strides�
9model_119/tf_op_layer_strided_slice_481/strided_slice_481StridedSlice(model_119/repeat_vector_59/Tile:output:0Hmodel_119/tf_op_layer_strided_slice_481/strided_slice_481/begin:output:0Fmodel_119/tf_op_layer_strided_slice_481/strided_slice_481/end:output:0Jmodel_119/tf_op_layer_strided_slice_481/strided_slice_481/strides:output:0*
Index0*
T0*
_cloned(*+
_output_shapes
:��������� *
ellipsis_mask2;
9model_119/tf_op_layer_strided_slice_481/strided_slice_481�
,model_119/dense_476/Tensordot/ReadVariableOpReadVariableOp5model_119_dense_476_tensordot_readvariableop_resource*
_output_shapes

: *
dtype02.
,model_119/dense_476/Tensordot/ReadVariableOp�
"model_119/dense_476/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2$
"model_119/dense_476/Tensordot/axes�
"model_119/dense_476/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2$
"model_119/dense_476/Tensordot/free�
#model_119/dense_476/Tensordot/ShapeShape)model_119/concatenate_179/concat:output:0*
T0*
_output_shapes
:2%
#model_119/dense_476/Tensordot/Shape�
+model_119/dense_476/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2-
+model_119/dense_476/Tensordot/GatherV2/axis�
&model_119/dense_476/Tensordot/GatherV2GatherV2,model_119/dense_476/Tensordot/Shape:output:0+model_119/dense_476/Tensordot/free:output:04model_119/dense_476/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2(
&model_119/dense_476/Tensordot/GatherV2�
-model_119/dense_476/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2/
-model_119/dense_476/Tensordot/GatherV2_1/axis�
(model_119/dense_476/Tensordot/GatherV2_1GatherV2,model_119/dense_476/Tensordot/Shape:output:0+model_119/dense_476/Tensordot/axes:output:06model_119/dense_476/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2*
(model_119/dense_476/Tensordot/GatherV2_1�
#model_119/dense_476/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2%
#model_119/dense_476/Tensordot/Const�
"model_119/dense_476/Tensordot/ProdProd/model_119/dense_476/Tensordot/GatherV2:output:0,model_119/dense_476/Tensordot/Const:output:0*
T0*
_output_shapes
: 2$
"model_119/dense_476/Tensordot/Prod�
%model_119/dense_476/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2'
%model_119/dense_476/Tensordot/Const_1�
$model_119/dense_476/Tensordot/Prod_1Prod1model_119/dense_476/Tensordot/GatherV2_1:output:0.model_119/dense_476/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2&
$model_119/dense_476/Tensordot/Prod_1�
)model_119/dense_476/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2+
)model_119/dense_476/Tensordot/concat/axis�
$model_119/dense_476/Tensordot/concatConcatV2+model_119/dense_476/Tensordot/free:output:0+model_119/dense_476/Tensordot/axes:output:02model_119/dense_476/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2&
$model_119/dense_476/Tensordot/concat�
#model_119/dense_476/Tensordot/stackPack+model_119/dense_476/Tensordot/Prod:output:0-model_119/dense_476/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2%
#model_119/dense_476/Tensordot/stack�
'model_119/dense_476/Tensordot/transpose	Transpose)model_119/concatenate_179/concat:output:0-model_119/dense_476/Tensordot/concat:output:0*
T0*+
_output_shapes
:��������� 2)
'model_119/dense_476/Tensordot/transpose�
%model_119/dense_476/Tensordot/ReshapeReshape+model_119/dense_476/Tensordot/transpose:y:0,model_119/dense_476/Tensordot/stack:output:0*
T0*0
_output_shapes
:������������������2'
%model_119/dense_476/Tensordot/Reshape�
$model_119/dense_476/Tensordot/MatMulMatMul.model_119/dense_476/Tensordot/Reshape:output:04model_119/dense_476/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� 2&
$model_119/dense_476/Tensordot/MatMul�
%model_119/dense_476/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: 2'
%model_119/dense_476/Tensordot/Const_2�
+model_119/dense_476/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2-
+model_119/dense_476/Tensordot/concat_1/axis�
&model_119/dense_476/Tensordot/concat_1ConcatV2/model_119/dense_476/Tensordot/GatherV2:output:0.model_119/dense_476/Tensordot/Const_2:output:04model_119/dense_476/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2(
&model_119/dense_476/Tensordot/concat_1�
model_119/dense_476/TensordotReshape.model_119/dense_476/Tensordot/MatMul:product:0/model_119/dense_476/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:���������  2
model_119/dense_476/Tensordot�
*model_119/dense_476/BiasAdd/ReadVariableOpReadVariableOp3model_119_dense_476_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02,
*model_119/dense_476/BiasAdd/ReadVariableOp�
model_119/dense_476/BiasAddAdd&model_119/dense_476/Tensordot:output:02model_119/dense_476/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������  2
model_119/dense_476/BiasAdd�
model_119/dense_476/ReluRelumodel_119/dense_476/BiasAdd:z:0*
T0*+
_output_shapes
:���������  2
model_119/dense_476/Relu�
)model_119/tf_op_layer_AddV2_119/AddV2_119AddV2&model_119/color_law/Tensordot:output:0Bmodel_119/tf_op_layer_strided_slice_481/strided_slice_481:output:0*
T0*
_cloned(*,
_output_shapes
:��������� �2+
)model_119/tf_op_layer_AddV2_119/AddV2_119�
,model_119/dense_477/Tensordot/ReadVariableOpReadVariableOp5model_119_dense_477_tensordot_readvariableop_resource*
_output_shapes
:	 �*
dtype02.
,model_119/dense_477/Tensordot/ReadVariableOp�
"model_119/dense_477/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2$
"model_119/dense_477/Tensordot/axes�
"model_119/dense_477/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2$
"model_119/dense_477/Tensordot/free�
#model_119/dense_477/Tensordot/ShapeShape&model_119/dense_476/Relu:activations:0*
T0*
_output_shapes
:2%
#model_119/dense_477/Tensordot/Shape�
+model_119/dense_477/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2-
+model_119/dense_477/Tensordot/GatherV2/axis�
&model_119/dense_477/Tensordot/GatherV2GatherV2,model_119/dense_477/Tensordot/Shape:output:0+model_119/dense_477/Tensordot/free:output:04model_119/dense_477/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2(
&model_119/dense_477/Tensordot/GatherV2�
-model_119/dense_477/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2/
-model_119/dense_477/Tensordot/GatherV2_1/axis�
(model_119/dense_477/Tensordot/GatherV2_1GatherV2,model_119/dense_477/Tensordot/Shape:output:0+model_119/dense_477/Tensordot/axes:output:06model_119/dense_477/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2*
(model_119/dense_477/Tensordot/GatherV2_1�
#model_119/dense_477/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2%
#model_119/dense_477/Tensordot/Const�
"model_119/dense_477/Tensordot/ProdProd/model_119/dense_477/Tensordot/GatherV2:output:0,model_119/dense_477/Tensordot/Const:output:0*
T0*
_output_shapes
: 2$
"model_119/dense_477/Tensordot/Prod�
%model_119/dense_477/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2'
%model_119/dense_477/Tensordot/Const_1�
$model_119/dense_477/Tensordot/Prod_1Prod1model_119/dense_477/Tensordot/GatherV2_1:output:0.model_119/dense_477/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2&
$model_119/dense_477/Tensordot/Prod_1�
)model_119/dense_477/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2+
)model_119/dense_477/Tensordot/concat/axis�
$model_119/dense_477/Tensordot/concatConcatV2+model_119/dense_477/Tensordot/free:output:0+model_119/dense_477/Tensordot/axes:output:02model_119/dense_477/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2&
$model_119/dense_477/Tensordot/concat�
#model_119/dense_477/Tensordot/stackPack+model_119/dense_477/Tensordot/Prod:output:0-model_119/dense_477/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2%
#model_119/dense_477/Tensordot/stack�
'model_119/dense_477/Tensordot/transpose	Transpose&model_119/dense_476/Relu:activations:0-model_119/dense_477/Tensordot/concat:output:0*
T0*+
_output_shapes
:���������  2)
'model_119/dense_477/Tensordot/transpose�
%model_119/dense_477/Tensordot/ReshapeReshape+model_119/dense_477/Tensordot/transpose:y:0,model_119/dense_477/Tensordot/stack:output:0*
T0*0
_output_shapes
:������������������2'
%model_119/dense_477/Tensordot/Reshape�
$model_119/dense_477/Tensordot/MatMulMatMul.model_119/dense_477/Tensordot/Reshape:output:04model_119/dense_477/Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2&
$model_119/dense_477/Tensordot/MatMul�
%model_119/dense_477/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:�2'
%model_119/dense_477/Tensordot/Const_2�
+model_119/dense_477/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2-
+model_119/dense_477/Tensordot/concat_1/axis�
&model_119/dense_477/Tensordot/concat_1ConcatV2/model_119/dense_477/Tensordot/GatherV2:output:0.model_119/dense_477/Tensordot/Const_2:output:04model_119/dense_477/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2(
&model_119/dense_477/Tensordot/concat_1�
model_119/dense_477/TensordotReshape.model_119/dense_477/Tensordot/MatMul:product:0/model_119/dense_477/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:��������� �2
model_119/dense_477/Tensordot�
*model_119/dense_477/BiasAdd/ReadVariableOpReadVariableOp3model_119_dense_477_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02,
*model_119/dense_477/BiasAdd/ReadVariableOp�
model_119/dense_477/BiasAddAdd&model_119/dense_477/Tensordot:output:02model_119/dense_477/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:��������� �2
model_119/dense_477/BiasAdd�
model_119/dense_477/ReluRelumodel_119/dense_477/BiasAdd:z:0*
T0*,
_output_shapes
:��������� �2
model_119/dense_477/Relu�
'model_119/tf_op_layer_Mul_354/Mul_354/xConst*
_output_shapes
: *
dtype0*
valueB
 *��̾2)
'model_119/tf_op_layer_Mul_354/Mul_354/x�
%model_119/tf_op_layer_Mul_354/Mul_354Mul0model_119/tf_op_layer_Mul_354/Mul_354/x:output:0-model_119/tf_op_layer_AddV2_119/AddV2_119:z:0*
T0*
_cloned(*,
_output_shapes
:��������� �2'
%model_119/tf_op_layer_Mul_354/Mul_354�
,model_119/dense_478/Tensordot/ReadVariableOpReadVariableOp5model_119_dense_478_tensordot_readvariableop_resource* 
_output_shapes
:
��*
dtype02.
,model_119/dense_478/Tensordot/ReadVariableOp�
"model_119/dense_478/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2$
"model_119/dense_478/Tensordot/axes�
"model_119/dense_478/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2$
"model_119/dense_478/Tensordot/free�
#model_119/dense_478/Tensordot/ShapeShape&model_119/dense_477/Relu:activations:0*
T0*
_output_shapes
:2%
#model_119/dense_478/Tensordot/Shape�
+model_119/dense_478/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2-
+model_119/dense_478/Tensordot/GatherV2/axis�
&model_119/dense_478/Tensordot/GatherV2GatherV2,model_119/dense_478/Tensordot/Shape:output:0+model_119/dense_478/Tensordot/free:output:04model_119/dense_478/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2(
&model_119/dense_478/Tensordot/GatherV2�
-model_119/dense_478/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2/
-model_119/dense_478/Tensordot/GatherV2_1/axis�
(model_119/dense_478/Tensordot/GatherV2_1GatherV2,model_119/dense_478/Tensordot/Shape:output:0+model_119/dense_478/Tensordot/axes:output:06model_119/dense_478/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2*
(model_119/dense_478/Tensordot/GatherV2_1�
#model_119/dense_478/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2%
#model_119/dense_478/Tensordot/Const�
"model_119/dense_478/Tensordot/ProdProd/model_119/dense_478/Tensordot/GatherV2:output:0,model_119/dense_478/Tensordot/Const:output:0*
T0*
_output_shapes
: 2$
"model_119/dense_478/Tensordot/Prod�
%model_119/dense_478/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2'
%model_119/dense_478/Tensordot/Const_1�
$model_119/dense_478/Tensordot/Prod_1Prod1model_119/dense_478/Tensordot/GatherV2_1:output:0.model_119/dense_478/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2&
$model_119/dense_478/Tensordot/Prod_1�
)model_119/dense_478/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2+
)model_119/dense_478/Tensordot/concat/axis�
$model_119/dense_478/Tensordot/concatConcatV2+model_119/dense_478/Tensordot/free:output:0+model_119/dense_478/Tensordot/axes:output:02model_119/dense_478/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2&
$model_119/dense_478/Tensordot/concat�
#model_119/dense_478/Tensordot/stackPack+model_119/dense_478/Tensordot/Prod:output:0-model_119/dense_478/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2%
#model_119/dense_478/Tensordot/stack�
'model_119/dense_478/Tensordot/transpose	Transpose&model_119/dense_477/Relu:activations:0-model_119/dense_478/Tensordot/concat:output:0*
T0*,
_output_shapes
:��������� �2)
'model_119/dense_478/Tensordot/transpose�
%model_119/dense_478/Tensordot/ReshapeReshape+model_119/dense_478/Tensordot/transpose:y:0,model_119/dense_478/Tensordot/stack:output:0*
T0*0
_output_shapes
:������������������2'
%model_119/dense_478/Tensordot/Reshape�
$model_119/dense_478/Tensordot/MatMulMatMul.model_119/dense_478/Tensordot/Reshape:output:04model_119/dense_478/Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2&
$model_119/dense_478/Tensordot/MatMul�
%model_119/dense_478/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:�2'
%model_119/dense_478/Tensordot/Const_2�
+model_119/dense_478/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2-
+model_119/dense_478/Tensordot/concat_1/axis�
&model_119/dense_478/Tensordot/concat_1ConcatV2/model_119/dense_478/Tensordot/GatherV2:output:0.model_119/dense_478/Tensordot/Const_2:output:04model_119/dense_478/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2(
&model_119/dense_478/Tensordot/concat_1�
model_119/dense_478/TensordotReshape.model_119/dense_478/Tensordot/MatMul:product:0/model_119/dense_478/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:��������� �2
model_119/dense_478/Tensordot�
*model_119/dense_478/BiasAdd/ReadVariableOpReadVariableOp3model_119_dense_478_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02,
*model_119/dense_478/BiasAdd/ReadVariableOp�
model_119/dense_478/BiasAddAdd&model_119/dense_478/Tensordot:output:02model_119/dense_478/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:��������� �2
model_119/dense_478/BiasAdd�
model_119/dense_478/ReluRelumodel_119/dense_478/BiasAdd:z:0*
T0*,
_output_shapes
:��������� �2
model_119/dense_478/Relu�
,model_119/dense_479/Tensordot/ReadVariableOpReadVariableOp5model_119_dense_479_tensordot_readvariableop_resource* 
_output_shapes
:
��*
dtype02.
,model_119/dense_479/Tensordot/ReadVariableOp�
"model_119/dense_479/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2$
"model_119/dense_479/Tensordot/axes�
"model_119/dense_479/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2$
"model_119/dense_479/Tensordot/free�
#model_119/dense_479/Tensordot/ShapeShape&model_119/dense_478/Relu:activations:0*
T0*
_output_shapes
:2%
#model_119/dense_479/Tensordot/Shape�
+model_119/dense_479/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2-
+model_119/dense_479/Tensordot/GatherV2/axis�
&model_119/dense_479/Tensordot/GatherV2GatherV2,model_119/dense_479/Tensordot/Shape:output:0+model_119/dense_479/Tensordot/free:output:04model_119/dense_479/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2(
&model_119/dense_479/Tensordot/GatherV2�
-model_119/dense_479/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2/
-model_119/dense_479/Tensordot/GatherV2_1/axis�
(model_119/dense_479/Tensordot/GatherV2_1GatherV2,model_119/dense_479/Tensordot/Shape:output:0+model_119/dense_479/Tensordot/axes:output:06model_119/dense_479/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2*
(model_119/dense_479/Tensordot/GatherV2_1�
#model_119/dense_479/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2%
#model_119/dense_479/Tensordot/Const�
"model_119/dense_479/Tensordot/ProdProd/model_119/dense_479/Tensordot/GatherV2:output:0,model_119/dense_479/Tensordot/Const:output:0*
T0*
_output_shapes
: 2$
"model_119/dense_479/Tensordot/Prod�
%model_119/dense_479/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2'
%model_119/dense_479/Tensordot/Const_1�
$model_119/dense_479/Tensordot/Prod_1Prod1model_119/dense_479/Tensordot/GatherV2_1:output:0.model_119/dense_479/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2&
$model_119/dense_479/Tensordot/Prod_1�
)model_119/dense_479/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2+
)model_119/dense_479/Tensordot/concat/axis�
$model_119/dense_479/Tensordot/concatConcatV2+model_119/dense_479/Tensordot/free:output:0+model_119/dense_479/Tensordot/axes:output:02model_119/dense_479/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2&
$model_119/dense_479/Tensordot/concat�
#model_119/dense_479/Tensordot/stackPack+model_119/dense_479/Tensordot/Prod:output:0-model_119/dense_479/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2%
#model_119/dense_479/Tensordot/stack�
'model_119/dense_479/Tensordot/transpose	Transpose&model_119/dense_478/Relu:activations:0-model_119/dense_479/Tensordot/concat:output:0*
T0*,
_output_shapes
:��������� �2)
'model_119/dense_479/Tensordot/transpose�
%model_119/dense_479/Tensordot/ReshapeReshape+model_119/dense_479/Tensordot/transpose:y:0,model_119/dense_479/Tensordot/stack:output:0*
T0*0
_output_shapes
:������������������2'
%model_119/dense_479/Tensordot/Reshape�
$model_119/dense_479/Tensordot/MatMulMatMul.model_119/dense_479/Tensordot/Reshape:output:04model_119/dense_479/Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2&
$model_119/dense_479/Tensordot/MatMul�
%model_119/dense_479/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:�2'
%model_119/dense_479/Tensordot/Const_2�
+model_119/dense_479/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2-
+model_119/dense_479/Tensordot/concat_1/axis�
&model_119/dense_479/Tensordot/concat_1ConcatV2/model_119/dense_479/Tensordot/GatherV2:output:0.model_119/dense_479/Tensordot/Const_2:output:04model_119/dense_479/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2(
&model_119/dense_479/Tensordot/concat_1�
model_119/dense_479/TensordotReshape.model_119/dense_479/Tensordot/MatMul:product:0/model_119/dense_479/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:��������� �2
model_119/dense_479/Tensordot�
*model_119/dense_479/BiasAdd/ReadVariableOpReadVariableOp3model_119_dense_479_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02,
*model_119/dense_479/BiasAdd/ReadVariableOp�
model_119/dense_479/BiasAddAdd&model_119/dense_479/Tensordot:output:02model_119/dense_479/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:��������� �2
model_119/dense_479/BiasAdd�
%model_119/tf_op_layer_Pow_59/Pow_59/xConst*
_output_shapes
: *
dtype0*
valueB
 *   A2'
%model_119/tf_op_layer_Pow_59/Pow_59/x�
#model_119/tf_op_layer_Pow_59/Pow_59Pow.model_119/tf_op_layer_Pow_59/Pow_59/x:output:0)model_119/tf_op_layer_Mul_354/Mul_354:z:0*
T0*
_cloned(*,
_output_shapes
:��������� �2%
#model_119/tf_op_layer_Pow_59/Pow_59�
%model_119/tf_op_layer_Mul_355/Mul_355Mulmodel_119/dense_479/BiasAdd:z:0'model_119/tf_op_layer_Pow_59/Pow_59:z:0*
T0*
_cloned(*,
_output_shapes
:��������� �2'
%model_119/tf_op_layer_Mul_355/Mul_355�
%model_119/tf_op_layer_Relu_55/Relu_55Relu)model_119/tf_op_layer_Mul_355/Mul_355:z:0*
T0*
_cloned(*,
_output_shapes
:��������� �2'
%model_119/tf_op_layer_Relu_55/Relu_55�
5model_119/tf_op_layer_Max_63/Max_63/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
���������27
5model_119/tf_op_layer_Max_63/Max_63/reduction_indices�
#model_119/tf_op_layer_Max_63/Max_63Max	input_240>model_119/tf_op_layer_Max_63/Max_63/reduction_indices:output:0*
T0*
_cloned(*+
_output_shapes
:��������� *
	keep_dims(2%
#model_119/tf_op_layer_Max_63/Max_63�
%model_119/tf_op_layer_Mul_356/Mul_356Mul3model_119/tf_op_layer_Relu_55/Relu_55:activations:0,model_119/tf_op_layer_Max_63/Max_63:output:0*
T0*
_cloned(*,
_output_shapes
:��������� �2'
%model_119/tf_op_layer_Mul_356/Mul_356�
IdentityIdentity)model_119/tf_op_layer_Mul_356/Mul_356:z:0*
T0*,
_output_shapes
:��������� �2

Identity"
identityIdentity:output:0*y
_input_shapesh
f:���������:��������� :��������� �::::::::::V R
'
_output_shapes
:���������
'
_user_specified_namelatent_params:_[
+
_output_shapes
:��������� 
,
_user_specified_nameconditional_params:WS
,
_output_shapes
:��������� �
#
_user_specified_name	input_240:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: 
�
�
E__inference_dense_477_layer_call_and_return_conditional_losses_461756

inputs%
!tensordot_readvariableop_resource#
biasadd_readvariableop_resource
identity��
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes
:	 �*
dtype02
Tensordot/ReadVariableOpj
Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
Tensordot/axesq
Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
Tensordot/freeX
Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:2
Tensordot/Shapet
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/GatherV2/axis�
Tensordot/GatherV2GatherV2Tensordot/Shape:output:0Tensordot/free:output:0 Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
Tensordot/GatherV2x
Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/GatherV2_1/axis�
Tensordot/GatherV2_1GatherV2Tensordot/Shape:output:0Tensordot/axes:output:0"Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
Tensordot/GatherV2_1l
Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot/Const�
Tensordot/ProdProdTensordot/GatherV2:output:0Tensordot/Const:output:0*
T0*
_output_shapes
: 2
Tensordot/Prodp
Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot/Const_1�
Tensordot/Prod_1ProdTensordot/GatherV2_1:output:0Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
Tensordot/Prod_1p
Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/concat/axis�
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat�
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
Tensordot/stack�
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*+
_output_shapes
:���������  2
Tensordot/transpose�
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:������������������2
Tensordot/Reshape�
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
Tensordot/MatMulq
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:�2
Tensordot/Const_2t
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/concat_1/axis�
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat_1�
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*,
_output_shapes
:��������� �2
	Tensordot�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02
BiasAdd/ReadVariableOp�
BiasAddAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:��������� �2	
BiasAddX
ReluReluBiasAdd:z:0*
T0*,
_output_shapes
:��������� �2
Reluk
IdentityIdentityRelu:activations:0*
T0*,
_output_shapes
:��������� �2

Identity"
identityIdentity:output:0*2
_input_shapes!
:���������  :::S O
+
_output_shapes
:���������  
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
�

*__inference_dense_478_layer_call_fn_462789

inputs
unknown
	unknown_0
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*,
_output_shapes
:��������� �*$
_read_only_resource_inputs
*-
config_proto

GPU

CPU2*0J 8*N
fIRG
E__inference_dense_478_layer_call_and_return_conditional_losses_4618172
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*,
_output_shapes
:��������� �2

Identity"
identityIdentity:output:0*3
_input_shapes"
 :��������� �::22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:��������� �
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
�
`
4__inference_tf_op_layer_Mul_356_layer_call_fn_462895
inputs_0
inputs_1
identity�
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*,
_output_shapes
:��������� �* 
_read_only_resource_inputs
 *-
config_proto

GPU

CPU2*0J 8*X
fSRQ
O__inference_tf_op_layer_Mul_356_layer_call_and_return_conditional_losses_4619412
PartitionedCallq
IdentityIdentityPartitionedCall:output:0*
T0*,
_output_shapes
:��������� �2

Identity"
identityIdentity:output:0*B
_input_shapes1
/:��������� �:��������� :V R
,
_output_shapes
:��������� �
"
_user_specified_name
inputs/0:UQ
+
_output_shapes
:��������� 
"
_user_specified_name
inputs/1
�
O
3__inference_tf_op_layer_Pow_59_layer_call_fn_462850

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*,
_output_shapes
:��������� �* 
_read_only_resource_inputs
 *-
config_proto

GPU

CPU2*0J 8*W
fRRP
N__inference_tf_op_layer_Pow_59_layer_call_and_return_conditional_losses_4618852
PartitionedCallq
IdentityIdentityPartitionedCall:output:0*
T0*,
_output_shapes
:��������� �2

Identity"
identityIdentity:output:0*+
_input_shapes
:��������� �:T P
,
_output_shapes
:��������� �
 
_user_specified_nameinputs
�
�
E__inference_dense_478_layer_call_and_return_conditional_losses_462780

inputs%
!tensordot_readvariableop_resource#
biasadd_readvariableop_resource
identity��
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource* 
_output_shapes
:
��*
dtype02
Tensordot/ReadVariableOpj
Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
Tensordot/axesq
Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
Tensordot/freeX
Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:2
Tensordot/Shapet
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/GatherV2/axis�
Tensordot/GatherV2GatherV2Tensordot/Shape:output:0Tensordot/free:output:0 Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
Tensordot/GatherV2x
Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/GatherV2_1/axis�
Tensordot/GatherV2_1GatherV2Tensordot/Shape:output:0Tensordot/axes:output:0"Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
Tensordot/GatherV2_1l
Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot/Const�
Tensordot/ProdProdTensordot/GatherV2:output:0Tensordot/Const:output:0*
T0*
_output_shapes
: 2
Tensordot/Prodp
Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot/Const_1�
Tensordot/Prod_1ProdTensordot/GatherV2_1:output:0Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
Tensordot/Prod_1p
Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/concat/axis�
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat�
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
Tensordot/stack�
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*,
_output_shapes
:��������� �2
Tensordot/transpose�
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:������������������2
Tensordot/Reshape�
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
Tensordot/MatMulq
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:�2
Tensordot/Const_2t
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/concat_1/axis�
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat_1�
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*,
_output_shapes
:��������� �2
	Tensordot�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02
BiasAdd/ReadVariableOp�
BiasAddAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:��������� �2	
BiasAddX
ReluReluBiasAdd:z:0*
T0*,
_output_shapes
:��������� �2
Reluk
IdentityIdentityRelu:activations:0*
T0*,
_output_shapes
:��������� �2

Identity"
identityIdentity:output:0*3
_input_shapes"
 :��������� �:::T P
,
_output_shapes
:��������� �
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
�
�
*__inference_model_119_layer_call_fn_462063
latent_params
conditional_params
	input_240
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCalllatent_paramsconditional_params	input_240unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7*
Tin
2*
Tout
2*,
_output_shapes
:��������� �*+
_read_only_resource_inputs
		
*-
config_proto

GPU

CPU2*0J 8*N
fIRG
E__inference_model_119_layer_call_and_return_conditional_losses_4620422
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*,
_output_shapes
:��������� �2

Identity"
identityIdentity:output:0*y
_input_shapesh
f:���������:��������� :��������� �:::::::::22
StatefulPartitionedCallStatefulPartitionedCall:V R
'
_output_shapes
:���������
'
_user_specified_namelatent_params:_[
+
_output_shapes
:��������� 
,
_user_specified_nameconditional_params:WS
,
_output_shapes
:��������� �
#
_user_specified_name	input_240:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: 
�
`
4__inference_tf_op_layer_Mul_355_layer_call_fn_462862
inputs_0
inputs_1
identity�
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*,
_output_shapes
:��������� �* 
_read_only_resource_inputs
 *-
config_proto

GPU

CPU2*0J 8*X
fSRQ
O__inference_tf_op_layer_Mul_355_layer_call_and_return_conditional_losses_4618992
PartitionedCallq
IdentityIdentityPartitionedCall:output:0*
T0*,
_output_shapes
:��������� �2

Identity"
identityIdentity:output:0*C
_input_shapes2
0:��������� �:��������� �:V R
,
_output_shapes
:��������� �
"
_user_specified_name
inputs/0:VR
,
_output_shapes
:��������� �
"
_user_specified_name
inputs/1
�
u
Y__inference_tf_op_layer_strided_slice_481_layer_call_and_return_conditional_losses_462692

inputs
identity�
strided_slice_481/beginConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_481/begin
strided_slice_481/endConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_481/end�
strided_slice_481/stridesConst*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_481/strides�
strided_slice_481StridedSliceinputs strided_slice_481/begin:output:0strided_slice_481/end:output:0"strided_slice_481/strides:output:0*
Index0*
T0*
_cloned(*+
_output_shapes
:��������� *
ellipsis_mask2
strided_slice_481r
IdentityIdentitystrided_slice_481:output:0*
T0*+
_output_shapes
:��������� 2

Identity"
identityIdentity:output:0**
_input_shapes
:��������� :S O
+
_output_shapes
:��������� 
 
_user_specified_nameinputs
�
�
E__inference_dense_479_layer_call_and_return_conditional_losses_462830

inputs%
!tensordot_readvariableop_resource#
biasadd_readvariableop_resource
identity��
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource* 
_output_shapes
:
��*
dtype02
Tensordot/ReadVariableOpj
Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
Tensordot/axesq
Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
Tensordot/freeX
Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:2
Tensordot/Shapet
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/GatherV2/axis�
Tensordot/GatherV2GatherV2Tensordot/Shape:output:0Tensordot/free:output:0 Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
Tensordot/GatherV2x
Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/GatherV2_1/axis�
Tensordot/GatherV2_1GatherV2Tensordot/Shape:output:0Tensordot/axes:output:0"Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
Tensordot/GatherV2_1l
Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot/Const�
Tensordot/ProdProdTensordot/GatherV2:output:0Tensordot/Const:output:0*
T0*
_output_shapes
: 2
Tensordot/Prodp
Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot/Const_1�
Tensordot/Prod_1ProdTensordot/GatherV2_1:output:0Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
Tensordot/Prod_1p
Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/concat/axis�
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat�
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
Tensordot/stack�
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*,
_output_shapes
:��������� �2
Tensordot/transpose�
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:������������������2
Tensordot/Reshape�
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
Tensordot/MatMulq
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:�2
Tensordot/Const_2t
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/concat_1/axis�
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat_1�
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*,
_output_shapes
:��������� �2
	Tensordot�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02
BiasAdd/ReadVariableOp�
BiasAddAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:��������� �2	
BiasAddd
IdentityIdentityBiasAdd:z:0*
T0*,
_output_shapes
:��������� �2

Identity"
identityIdentity:output:0*3
_input_shapes"
 :��������� �:::T P
,
_output_shapes
:��������� �
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: "�L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*�
serving_default�
U
conditional_params?
$serving_default_conditional_params:0��������� 
D
	input_2407
serving_default_input_240:0��������� �
G
latent_params6
serving_default_latent_params:0���������L
tf_op_layer_Mul_3565
StatefulPartitionedCall:0��������� �tensorflow/serving/predict:�
��
layer-0
layer-1
layer-2
layer-3
layer-4
layer-5
layer-6
layer-7
	layer_with_weights-0
	layer-8

layer_with_weights-1

layer-9
layer-10
layer_with_weights-2
layer-11
layer-12
layer_with_weights-3
layer-13
layer-14
layer_with_weights-4
layer-15
layer-16
layer-17
layer-18
layer-19
layer-20
layer-21
	variables
regularization_losses
trainable_variables
	keras_api

signatures
�_default_save_signature
+�&call_and_return_all_conditional_losses
�__call__"��
_tf_keras_model��{"class_name": "Model", "name": "model_119", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "model_119", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 6]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "latent_params"}, "name": "latent_params", "inbound_nodes": []}, {"class_name": "RepeatVector", "config": {"name": "repeat_vector_59", "trainable": true, "dtype": "float32", "n": 32}, "name": "repeat_vector_59", "inbound_nodes": [[["latent_params", 0, 0, {}]]]}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 32, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "conditional_params"}, "name": "conditional_params", "inbound_nodes": []}, {"class_name": "TensorFlowOpLayer", "config": {"name": "strided_slice_480", "trainable": true, "dtype": "float32", "node_def": {"name": "strided_slice_480", "op": "StridedSlice", "input": ["repeat_vector_59/Identity", "strided_slice_480/begin", "strided_slice_480/end", "strided_slice_480/strides"], "attr": {"end_mask": {"i": "0"}, "begin_mask": {"i": "0"}, "shrink_axis_mask": {"i": "0"}, "new_axis_mask": {"i": "0"}, "T": {"type": "DT_FLOAT"}, "ellipsis_mask": {"i": "1"}, "Index": {"type": "DT_INT32"}}}, "constants": {"1": [0, 0], "2": [0, 1], "3": [1, 1]}}, "name": "tf_op_layer_strided_slice_480", "inbound_nodes": [[["repeat_vector_59", 0, 0, {}]]]}, {"class_name": "TensorFlowOpLayer", "config": {"name": "strided_slice_483", "trainable": true, "dtype": "float32", "node_def": {"name": "strided_slice_483", "op": "StridedSlice", "input": ["repeat_vector_59/Identity", "strided_slice_483/begin", "strided_slice_483/end", "strided_slice_483/strides"], "attr": {"new_axis_mask": {"i": "0"}, "begin_mask": {"i": "0"}, "end_mask": {"i": "2"}, "Index": {"type": "DT_INT32"}, "ellipsis_mask": {"i": "1"}, "shrink_axis_mask": {"i": "0"}, "T": {"type": "DT_FLOAT"}}}, "constants": {"1": [0, 3], "2": [0, 0], "3": [1, 1]}}, "name": "tf_op_layer_strided_slice_483", "inbound_nodes": [[["repeat_vector_59", 0, 0, {}]]]}, {"class_name": "TensorFlowOpLayer", "config": {"name": "AddV2_118", "trainable": true, "dtype": "float32", "node_def": {"name": "AddV2_118", "op": "AddV2", "input": ["conditional_params_62", "strided_slice_480"], "attr": {"T": {"type": "DT_FLOAT"}}}, "constants": {}}, "name": "tf_op_layer_AddV2_118", "inbound_nodes": [[["conditional_params", 0, 0, {}], ["tf_op_layer_strided_slice_480", 0, 0, {}]]]}, {"class_name": "Concatenate", "config": {"name": "concatenate_179", "trainable": true, "dtype": "float32", "axis": -1}, "name": "concatenate_179", "inbound_nodes": [[["tf_op_layer_strided_slice_483", 0, 0, {}], ["tf_op_layer_AddV2_118", 0, 0, {}]]]}, {"class_name": "TensorFlowOpLayer", "config": {"name": "strided_slice_482", "trainable": true, "dtype": "float32", "node_def": {"name": "strided_slice_482", "op": "StridedSlice", "input": ["repeat_vector_59/Identity", "strided_slice_482/begin", "strided_slice_482/end", "strided_slice_482/strides"], "attr": {"ellipsis_mask": {"i": "1"}, "shrink_axis_mask": {"i": "0"}, "T": {"type": "DT_FLOAT"}, "begin_mask": {"i": "0"}, "Index": {"type": "DT_INT32"}, "new_axis_mask": {"i": "0"}, "end_mask": {"i": "0"}}}, "constants": {"1": [0, 2], "2": [0, 3], "3": [1, 1]}}, "name": "tf_op_layer_strided_slice_482", "inbound_nodes": [[["repeat_vector_59", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_476", "trainable": true, "dtype": "float32", "units": 32, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_476", "inbound_nodes": [[["concatenate_179", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "color_law", "trainable": false, "dtype": "float32", "units": 288, "activation": "linear", "use_bias": false, "kernel_initializer": {"class_name": "Constant", "config": {"value": [1.733986283286547, 1.7287693811029068, 1.7235902690277825, 1.7184478730039496, 1.713341136989258, 1.7082690226720982, 1.703230509249906, 1.698224593198338, 1.693250288043143, 1.688306624134713, 1.6833926484252757, 1.6785074242486995, 1.673650031102894, 1.6688195644347634, 1.664015135427689, 1.659235870791517, 1.6544809125550193, 1.6497494178608025, 1.6450405587626389, 1.6403535220251868, 1.635687508926083, 1.631041735060371, 1.626415430147253, 1.6218078378391225, 1.6172182155328652, 1.6126458341834013, 1.6080899781194373, 1.603549944861411, 1.5990250449416044, 1.5945146017263927, 1.5900179512406165, 1.5855344419940425, 1.5810634348099024, 1.5766043026554741, 1.572156430474695, 1.567719215022769, 1.5632920647027608, 1.5588743994041487, 1.554465650343312, 1.5500652599059337, 1.5456726814912982, 1.5412873793584616, 1.5369088284742718, 1.532536514363226, 1.52816993295913, 1.5238085904585568, 1.5194520031760688, 1.5150996974011943, 1.510751209257138, 1.506406084561195, 1.5020638786868683, 1.4977241564276482, 1.4933864918624562, 1.4890504682227241, 1.4847156777610842, 1.4803817216216668, 1.4760482097119774, 1.471714760576336, 1.4673810012708666, 1.4630465672400124, 1.458711102194566, 1.4543742579911925, 1.4500356945134296, 1.4456950795541552, 1.441352088699493, 1.4370064052141502, 1.4326577170773538, 1.4283052994820302, 1.42394758628978, 1.4195829394917132, 1.4152097553571679, 1.4108264639503714, 1.4064315286529432, 1.4020234456921643, 1.3976007436749458, 1.3931619831274442, 1.3887057560402456, 1.384230685419072, 1.379735424840937, 1.3752186580156902, 1.3706790983529, 1.3661154885340008, 1.3615266000896553, 1.3569112329822757, 1.3522682151936312, 1.3475964023175002, 1.342894677157307, 1.3381619493286763, 1.3333971548668704, 1.3285992558390292, 1.323767239961183, 1.318900120219965, 1.3139969344989837, 1.3090567452097988, 1.3040786389274424, 1.2990617260304427, 1.2940051403452952, 1.288908038795336, 1.2837696010539528, 1.2785890292021074, 1.2733655473901047, 1.2680984015035532, 1.2627868588334983, 1.257430207750652, 1.2520277573836838, 1.246579346471573, 1.2410871323811035, 1.2355538901996845, 1.2299823462385933, 1.2243751786311778, 1.2187350179713072, 1.2130644479443282, 1.2073660059506393, 1.201642183721952, 1.195895427930307, 1.1901281407899522, 1.1843426806521349, 1.1785413625929024, 1.1727264589939799, 1.166900200116804, 1.1610647746697969, 1.1552223303689295, 1.1493749744916821, 1.1435247744244514, 1.1376737582034773, 1.1318239150493759, 1.1259771958953362, 1.1201355139090525, 1.1143007450084679, 1.1084747283713863, 1.1026592669390394, 1.096856127913651, 1.0910670432500833, 1.0852937101416318, 1.0795377915000135, 1.0738009164296394, 1.0680846806962128, 1.0623906471897342, 1.056720346381954, 1.05107527677836, 1.0454569053647416, 1.0398666680483932, 1.034305970094022, 1.0287761865544245, 1.0232786626959696, 1.017814714418972, 1.0123856286729889, 1.0069926638671147, 1.0016370502753202, 0.9963199904368925, 0.9910426595520385, 0.9858062058726873, 0.9806116792336146, 0.975458939804804, 0.9703469032967426, 0.965274475843901, 0.9602405815868156, 0.9552441624369885, 0.9502841778445262, 0.9453596045684957, 0.9404694364499552, 0.935612684187638, 0.9307883751162611, 0.9259955529874249, 0.9212332777530796, 0.9165006253515322, 0.9117966874959529, 0.9071205714653741, 0.9024713998981359, 0.8978483105877624, 0.8932504562812408, 0.8886770044796674, 0.8841271372412491, 0.8796000509866208, 0.8750949563064638, 0.8706110777713957, 0.8661476537441003, 0.8617039361936851, 0.8572791905122311, 0.8528726953335185, 0.8484838197847225, 0.8441124032870609, 0.8397584556982287, 0.8354219857892379, 0.831103000996631, 0.8268015074441755, 0.822517509964287, 0.8182510121191809, 0.814002016221756, 0.8097705233562216, 0.8055565333984581, 0.8013600450361216, 0.7971810557885, 0.7930195620261065, 0.7888755589900335, 0.7847490408110609, 0.7806400005285148, 0.7765484301088936, 0.7724743204642488, 0.7684176614703389, 0.7643784419845474, 0.7603566498635708, 0.7563522719808821, 0.7523652942439681, 0.7483957016113454, 0.7444434781093592, 0.7405086068487651, 0.7365910700410948, 0.7326908490148105, 0.7288079242312535, 0.7249422753003828, 0.7210938809963111, 0.7172627192726369, 0.7134487672775773, 0.7096520013689066, 0.705872397128695, 0.7021099293778591, 0.6983645721905184, 0.6946362989081626, 0.6909250821536369, 0.6872308938449425, 0.6835537052088484, 0.6798934867943335, 0.6762502084858424, 0.6726238395163708, 0.6690143484803764, 0.6654217033465153, 0.661845871470212, 0.658286819606059, 0.6547445139200541, 0.6512189200016723, 0.6477100028757729, 0.6442177270143525, 0.6407420563481341, 0.6372829542780033, 0.6338403836862915, 0.6304143069479011, 0.6270046859412867, 0.6236114820592785, 0.6202346562197689, 0.6168741688762457, 0.6135299800281842, 0.6102020492312966, 0.6068903356076423, 0.6035947978555963, 0.6003153942596847, 0.5970520827002805, 0.5938048206631684, 0.5905735652489754, 0.5873582731824692, 0.5841589008217343, 0.5809754041672106, 0.5778077388706131, 0.5746558602437234, 0.5715197232670574, 0.5683992825984165, 0.5652944925813117, 0.5622053072532722, 0.5591316803540352, 0.5560735653336232, 0.5530309153603018, 0.5500036833284264, 0.5469918218661778, 0.5439952833431848, 0.5410140198780397, 0.5380479833457052, 0.5350971253848144, 0.5321613974048656, 0.5292407505933108, 0.5263351359225454, 0.523444504156795, 0.5205688058588979, 0.5177079913969939, 0.5148620109511113, 0.5120308145196603, 0.5092143519258254, 0.5064125728238699, 0.5036254267053438, 0.5008528629051977, 0.49809483060780846, 0.49535127885291513, 0.4926221565414631, 0.489907412441364, 0.48720699519316507, 0.48452085331563677, 0.4818489352112722, 0.47919118916554737, 0.4765475633769238]}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "color_law", "inbound_nodes": [[["tf_op_layer_strided_slice_482", 0, 0, {}]]]}, {"class_name": "TensorFlowOpLayer", "config": {"name": "strided_slice_481", "trainable": true, "dtype": "float32", "node_def": {"name": "strided_slice_481", "op": "StridedSlice", "input": ["repeat_vector_59/Identity", "strided_slice_481/begin", "strided_slice_481/end", "strided_slice_481/strides"], "attr": {"ellipsis_mask": {"i": "1"}, "T": {"type": "DT_FLOAT"}, "Index": {"type": "DT_INT32"}, "shrink_axis_mask": {"i": "0"}, "begin_mask": {"i": "0"}, "new_axis_mask": {"i": "0"}, "end_mask": {"i": "0"}}}, "constants": {"1": [0, 1], "2": [0, 2], "3": [1, 1]}}, "name": "tf_op_layer_strided_slice_481", "inbound_nodes": [[["repeat_vector_59", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_477", "trainable": true, "dtype": "float32", "units": 128, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_477", "inbound_nodes": [[["dense_476", 0, 0, {}]]]}, {"class_name": "TensorFlowOpLayer", "config": {"name": "AddV2_119", "trainable": true, "dtype": "float32", "node_def": {"name": "AddV2_119", "op": "AddV2", "input": ["color_law_62/Identity", "strided_slice_481"], "attr": {"T": {"type": "DT_FLOAT"}}}, "constants": {}}, "name": "tf_op_layer_AddV2_119", "inbound_nodes": [[["color_law", 0, 0, {}], ["tf_op_layer_strided_slice_481", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_478", "trainable": true, "dtype": "float32", "units": 256, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_478", "inbound_nodes": [[["dense_477", 0, 0, {}]]]}, {"class_name": "TensorFlowOpLayer", "config": {"name": "Mul_354", "trainable": true, "dtype": "float32", "node_def": {"name": "Mul_354", "op": "Mul", "input": ["Mul_354/x", "AddV2_119"], "attr": {"T": {"type": "DT_FLOAT"}}}, "constants": {"0": -0.4000000059604645}}, "name": "tf_op_layer_Mul_354", "inbound_nodes": [[["tf_op_layer_AddV2_119", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_479", "trainable": true, "dtype": "float32", "units": 288, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_479", "inbound_nodes": [[["dense_478", 0, 0, {}]]]}, {"class_name": "TensorFlowOpLayer", "config": {"name": "Pow_59", "trainable": true, "dtype": "float32", "node_def": {"name": "Pow_59", "op": "Pow", "input": ["Pow_59/x", "Mul_354"], "attr": {"T": {"type": "DT_FLOAT"}}}, "constants": {"0": 10.0}}, "name": "tf_op_layer_Pow_59", "inbound_nodes": [[["tf_op_layer_Mul_354", 0, 0, {}]]]}, {"class_name": "TensorFlowOpLayer", "config": {"name": "Mul_355", "trainable": true, "dtype": "float32", "node_def": {"name": "Mul_355", "op": "Mul", "input": ["dense_479/Identity", "Pow_59"], "attr": {"T": {"type": "DT_FLOAT"}}}, "constants": {}}, "name": "tf_op_layer_Mul_355", "inbound_nodes": [[["dense_479", 0, 0, {}], ["tf_op_layer_Pow_59", 0, 0, {}]]]}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 32, 288]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_240"}, "name": "input_240", "inbound_nodes": []}, {"class_name": "TensorFlowOpLayer", "config": {"name": "Relu_55", "trainable": true, "dtype": "float32", "node_def": {"name": "Relu_55", "op": "Relu", "input": ["Mul_355"], "attr": {"T": {"type": "DT_FLOAT"}}}, "constants": {}}, "name": "tf_op_layer_Relu_55", "inbound_nodes": [[["tf_op_layer_Mul_355", 0, 0, {}]]]}, {"class_name": "TensorFlowOpLayer", "config": {"name": "Max_63", "trainable": true, "dtype": "float32", "node_def": {"name": "Max_63", "op": "Max", "input": ["input_240", "Max_63/reduction_indices"], "attr": {"T": {"type": "DT_FLOAT"}, "keep_dims": {"b": true}, "Tidx": {"type": "DT_INT32"}}}, "constants": {"1": -1}}, "name": "tf_op_layer_Max_63", "inbound_nodes": [[["input_240", 0, 0, {}]]]}, {"class_name": "TensorFlowOpLayer", "config": {"name": "Mul_356", "trainable": true, "dtype": "float32", "node_def": {"name": "Mul_356", "op": "Mul", "input": ["Relu_55", "Max_63"], "attr": {"T": {"type": "DT_FLOAT"}}}, "constants": {}}, "name": "tf_op_layer_Mul_356", "inbound_nodes": [[["tf_op_layer_Relu_55", 0, 0, {}], ["tf_op_layer_Max_63", 0, 0, {}]]]}], "input_layers": [["latent_params", 0, 0], ["conditional_params", 0, 0], ["input_240", 0, 0]], "output_layers": [["tf_op_layer_Mul_356", 0, 0]]}, "build_input_shape": [{"class_name": "TensorShape", "items": [null, 6]}, {"class_name": "TensorShape", "items": [null, 32, 1]}, {"class_name": "TensorShape", "items": [null, 32, 288]}], "is_graph_network": true, "keras_version": "2.3.0-tf", "backend": "tensorflow", "model_config": {"class_name": "Model", "config": {"name": "model_119", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 6]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "latent_params"}, "name": "latent_params", "inbound_nodes": []}, {"class_name": "RepeatVector", "config": {"name": "repeat_vector_59", "trainable": true, "dtype": "float32", "n": 32}, "name": "repeat_vector_59", "inbound_nodes": [[["latent_params", 0, 0, {}]]]}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 32, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "conditional_params"}, "name": "conditional_params", "inbound_nodes": []}, {"class_name": "TensorFlowOpLayer", "config": {"name": "strided_slice_480", "trainable": true, "dtype": "float32", "node_def": {"name": "strided_slice_480", "op": "StridedSlice", "input": ["repeat_vector_59/Identity", "strided_slice_480/begin", "strided_slice_480/end", "strided_slice_480/strides"], "attr": {"end_mask": {"i": "0"}, "begin_mask": {"i": "0"}, "shrink_axis_mask": {"i": "0"}, "new_axis_mask": {"i": "0"}, "T": {"type": "DT_FLOAT"}, "ellipsis_mask": {"i": "1"}, "Index": {"type": "DT_INT32"}}}, "constants": {"1": [0, 0], "2": [0, 1], "3": [1, 1]}}, "name": "tf_op_layer_strided_slice_480", "inbound_nodes": [[["repeat_vector_59", 0, 0, {}]]]}, {"class_name": "TensorFlowOpLayer", "config": {"name": "strided_slice_483", "trainable": true, "dtype": "float32", "node_def": {"name": "strided_slice_483", "op": "StridedSlice", "input": ["repeat_vector_59/Identity", "strided_slice_483/begin", "strided_slice_483/end", "strided_slice_483/strides"], "attr": {"new_axis_mask": {"i": "0"}, "begin_mask": {"i": "0"}, "end_mask": {"i": "2"}, "Index": {"type": "DT_INT32"}, "ellipsis_mask": {"i": "1"}, "shrink_axis_mask": {"i": "0"}, "T": {"type": "DT_FLOAT"}}}, "constants": {"1": [0, 3], "2": [0, 0], "3": [1, 1]}}, "name": "tf_op_layer_strided_slice_483", "inbound_nodes": [[["repeat_vector_59", 0, 0, {}]]]}, {"class_name": "TensorFlowOpLayer", "config": {"name": "AddV2_118", "trainable": true, "dtype": "float32", "node_def": {"name": "AddV2_118", "op": "AddV2", "input": ["conditional_params_62", "strided_slice_480"], "attr": {"T": {"type": "DT_FLOAT"}}}, "constants": {}}, "name": "tf_op_layer_AddV2_118", "inbound_nodes": [[["conditional_params", 0, 0, {}], ["tf_op_layer_strided_slice_480", 0, 0, {}]]]}, {"class_name": "Concatenate", "config": {"name": "concatenate_179", "trainable": true, "dtype": "float32", "axis": -1}, "name": "concatenate_179", "inbound_nodes": [[["tf_op_layer_strided_slice_483", 0, 0, {}], ["tf_op_layer_AddV2_118", 0, 0, {}]]]}, {"class_name": "TensorFlowOpLayer", "config": {"name": "strided_slice_482", "trainable": true, "dtype": "float32", "node_def": {"name": "strided_slice_482", "op": "StridedSlice", "input": ["repeat_vector_59/Identity", "strided_slice_482/begin", "strided_slice_482/end", "strided_slice_482/strides"], "attr": {"ellipsis_mask": {"i": "1"}, "shrink_axis_mask": {"i": "0"}, "T": {"type": "DT_FLOAT"}, "begin_mask": {"i": "0"}, "Index": {"type": "DT_INT32"}, "new_axis_mask": {"i": "0"}, "end_mask": {"i": "0"}}}, "constants": {"1": [0, 2], "2": [0, 3], "3": [1, 1]}}, "name": "tf_op_layer_strided_slice_482", "inbound_nodes": [[["repeat_vector_59", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_476", "trainable": true, "dtype": "float32", "units": 32, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_476", "inbound_nodes": [[["concatenate_179", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "color_law", "trainable": false, "dtype": "float32", "units": 288, "activation": "linear", "use_bias": false, "kernel_initializer": {"class_name": "Constant", "config": {"value": [1.733986283286547, 1.7287693811029068, 1.7235902690277825, 1.7184478730039496, 1.713341136989258, 1.7082690226720982, 1.703230509249906, 1.698224593198338, 1.693250288043143, 1.688306624134713, 1.6833926484252757, 1.6785074242486995, 1.673650031102894, 1.6688195644347634, 1.664015135427689, 1.659235870791517, 1.6544809125550193, 1.6497494178608025, 1.6450405587626389, 1.6403535220251868, 1.635687508926083, 1.631041735060371, 1.626415430147253, 1.6218078378391225, 1.6172182155328652, 1.6126458341834013, 1.6080899781194373, 1.603549944861411, 1.5990250449416044, 1.5945146017263927, 1.5900179512406165, 1.5855344419940425, 1.5810634348099024, 1.5766043026554741, 1.572156430474695, 1.567719215022769, 1.5632920647027608, 1.5588743994041487, 1.554465650343312, 1.5500652599059337, 1.5456726814912982, 1.5412873793584616, 1.5369088284742718, 1.532536514363226, 1.52816993295913, 1.5238085904585568, 1.5194520031760688, 1.5150996974011943, 1.510751209257138, 1.506406084561195, 1.5020638786868683, 1.4977241564276482, 1.4933864918624562, 1.4890504682227241, 1.4847156777610842, 1.4803817216216668, 1.4760482097119774, 1.471714760576336, 1.4673810012708666, 1.4630465672400124, 1.458711102194566, 1.4543742579911925, 1.4500356945134296, 1.4456950795541552, 1.441352088699493, 1.4370064052141502, 1.4326577170773538, 1.4283052994820302, 1.42394758628978, 1.4195829394917132, 1.4152097553571679, 1.4108264639503714, 1.4064315286529432, 1.4020234456921643, 1.3976007436749458, 1.3931619831274442, 1.3887057560402456, 1.384230685419072, 1.379735424840937, 1.3752186580156902, 1.3706790983529, 1.3661154885340008, 1.3615266000896553, 1.3569112329822757, 1.3522682151936312, 1.3475964023175002, 1.342894677157307, 1.3381619493286763, 1.3333971548668704, 1.3285992558390292, 1.323767239961183, 1.318900120219965, 1.3139969344989837, 1.3090567452097988, 1.3040786389274424, 1.2990617260304427, 1.2940051403452952, 1.288908038795336, 1.2837696010539528, 1.2785890292021074, 1.2733655473901047, 1.2680984015035532, 1.2627868588334983, 1.257430207750652, 1.2520277573836838, 1.246579346471573, 1.2410871323811035, 1.2355538901996845, 1.2299823462385933, 1.2243751786311778, 1.2187350179713072, 1.2130644479443282, 1.2073660059506393, 1.201642183721952, 1.195895427930307, 1.1901281407899522, 1.1843426806521349, 1.1785413625929024, 1.1727264589939799, 1.166900200116804, 1.1610647746697969, 1.1552223303689295, 1.1493749744916821, 1.1435247744244514, 1.1376737582034773, 1.1318239150493759, 1.1259771958953362, 1.1201355139090525, 1.1143007450084679, 1.1084747283713863, 1.1026592669390394, 1.096856127913651, 1.0910670432500833, 1.0852937101416318, 1.0795377915000135, 1.0738009164296394, 1.0680846806962128, 1.0623906471897342, 1.056720346381954, 1.05107527677836, 1.0454569053647416, 1.0398666680483932, 1.034305970094022, 1.0287761865544245, 1.0232786626959696, 1.017814714418972, 1.0123856286729889, 1.0069926638671147, 1.0016370502753202, 0.9963199904368925, 0.9910426595520385, 0.9858062058726873, 0.9806116792336146, 0.975458939804804, 0.9703469032967426, 0.965274475843901, 0.9602405815868156, 0.9552441624369885, 0.9502841778445262, 0.9453596045684957, 0.9404694364499552, 0.935612684187638, 0.9307883751162611, 0.9259955529874249, 0.9212332777530796, 0.9165006253515322, 0.9117966874959529, 0.9071205714653741, 0.9024713998981359, 0.8978483105877624, 0.8932504562812408, 0.8886770044796674, 0.8841271372412491, 0.8796000509866208, 0.8750949563064638, 0.8706110777713957, 0.8661476537441003, 0.8617039361936851, 0.8572791905122311, 0.8528726953335185, 0.8484838197847225, 0.8441124032870609, 0.8397584556982287, 0.8354219857892379, 0.831103000996631, 0.8268015074441755, 0.822517509964287, 0.8182510121191809, 0.814002016221756, 0.8097705233562216, 0.8055565333984581, 0.8013600450361216, 0.7971810557885, 0.7930195620261065, 0.7888755589900335, 0.7847490408110609, 0.7806400005285148, 0.7765484301088936, 0.7724743204642488, 0.7684176614703389, 0.7643784419845474, 0.7603566498635708, 0.7563522719808821, 0.7523652942439681, 0.7483957016113454, 0.7444434781093592, 0.7405086068487651, 0.7365910700410948, 0.7326908490148105, 0.7288079242312535, 0.7249422753003828, 0.7210938809963111, 0.7172627192726369, 0.7134487672775773, 0.7096520013689066, 0.705872397128695, 0.7021099293778591, 0.6983645721905184, 0.6946362989081626, 0.6909250821536369, 0.6872308938449425, 0.6835537052088484, 0.6798934867943335, 0.6762502084858424, 0.6726238395163708, 0.6690143484803764, 0.6654217033465153, 0.661845871470212, 0.658286819606059, 0.6547445139200541, 0.6512189200016723, 0.6477100028757729, 0.6442177270143525, 0.6407420563481341, 0.6372829542780033, 0.6338403836862915, 0.6304143069479011, 0.6270046859412867, 0.6236114820592785, 0.6202346562197689, 0.6168741688762457, 0.6135299800281842, 0.6102020492312966, 0.6068903356076423, 0.6035947978555963, 0.6003153942596847, 0.5970520827002805, 0.5938048206631684, 0.5905735652489754, 0.5873582731824692, 0.5841589008217343, 0.5809754041672106, 0.5778077388706131, 0.5746558602437234, 0.5715197232670574, 0.5683992825984165, 0.5652944925813117, 0.5622053072532722, 0.5591316803540352, 0.5560735653336232, 0.5530309153603018, 0.5500036833284264, 0.5469918218661778, 0.5439952833431848, 0.5410140198780397, 0.5380479833457052, 0.5350971253848144, 0.5321613974048656, 0.5292407505933108, 0.5263351359225454, 0.523444504156795, 0.5205688058588979, 0.5177079913969939, 0.5148620109511113, 0.5120308145196603, 0.5092143519258254, 0.5064125728238699, 0.5036254267053438, 0.5008528629051977, 0.49809483060780846, 0.49535127885291513, 0.4926221565414631, 0.489907412441364, 0.48720699519316507, 0.48452085331563677, 0.4818489352112722, 0.47919118916554737, 0.4765475633769238]}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "color_law", "inbound_nodes": [[["tf_op_layer_strided_slice_482", 0, 0, {}]]]}, {"class_name": "TensorFlowOpLayer", "config": {"name": "strided_slice_481", "trainable": true, "dtype": "float32", "node_def": {"name": "strided_slice_481", "op": "StridedSlice", "input": ["repeat_vector_59/Identity", "strided_slice_481/begin", "strided_slice_481/end", "strided_slice_481/strides"], "attr": {"ellipsis_mask": {"i": "1"}, "T": {"type": "DT_FLOAT"}, "Index": {"type": "DT_INT32"}, "shrink_axis_mask": {"i": "0"}, "begin_mask": {"i": "0"}, "new_axis_mask": {"i": "0"}, "end_mask": {"i": "0"}}}, "constants": {"1": [0, 1], "2": [0, 2], "3": [1, 1]}}, "name": "tf_op_layer_strided_slice_481", "inbound_nodes": [[["repeat_vector_59", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_477", "trainable": true, "dtype": "float32", "units": 128, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_477", "inbound_nodes": [[["dense_476", 0, 0, {}]]]}, {"class_name": "TensorFlowOpLayer", "config": {"name": "AddV2_119", "trainable": true, "dtype": "float32", "node_def": {"name": "AddV2_119", "op": "AddV2", "input": ["color_law_62/Identity", "strided_slice_481"], "attr": {"T": {"type": "DT_FLOAT"}}}, "constants": {}}, "name": "tf_op_layer_AddV2_119", "inbound_nodes": [[["color_law", 0, 0, {}], ["tf_op_layer_strided_slice_481", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_478", "trainable": true, "dtype": "float32", "units": 256, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_478", "inbound_nodes": [[["dense_477", 0, 0, {}]]]}, {"class_name": "TensorFlowOpLayer", "config": {"name": "Mul_354", "trainable": true, "dtype": "float32", "node_def": {"name": "Mul_354", "op": "Mul", "input": ["Mul_354/x", "AddV2_119"], "attr": {"T": {"type": "DT_FLOAT"}}}, "constants": {"0": -0.4000000059604645}}, "name": "tf_op_layer_Mul_354", "inbound_nodes": [[["tf_op_layer_AddV2_119", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_479", "trainable": true, "dtype": "float32", "units": 288, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_479", "inbound_nodes": [[["dense_478", 0, 0, {}]]]}, {"class_name": "TensorFlowOpLayer", "config": {"name": "Pow_59", "trainable": true, "dtype": "float32", "node_def": {"name": "Pow_59", "op": "Pow", "input": ["Pow_59/x", "Mul_354"], "attr": {"T": {"type": "DT_FLOAT"}}}, "constants": {"0": 10.0}}, "name": "tf_op_layer_Pow_59", "inbound_nodes": [[["tf_op_layer_Mul_354", 0, 0, {}]]]}, {"class_name": "TensorFlowOpLayer", "config": {"name": "Mul_355", "trainable": true, "dtype": "float32", "node_def": {"name": "Mul_355", "op": "Mul", "input": ["dense_479/Identity", "Pow_59"], "attr": {"T": {"type": "DT_FLOAT"}}}, "constants": {}}, "name": "tf_op_layer_Mul_355", "inbound_nodes": [[["dense_479", 0, 0, {}], ["tf_op_layer_Pow_59", 0, 0, {}]]]}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 32, 288]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_240"}, "name": "input_240", "inbound_nodes": []}, {"class_name": "TensorFlowOpLayer", "config": {"name": "Relu_55", "trainable": true, "dtype": "float32", "node_def": {"name": "Relu_55", "op": "Relu", "input": ["Mul_355"], "attr": {"T": {"type": "DT_FLOAT"}}}, "constants": {}}, "name": "tf_op_layer_Relu_55", "inbound_nodes": [[["tf_op_layer_Mul_355", 0, 0, {}]]]}, {"class_name": "TensorFlowOpLayer", "config": {"name": "Max_63", "trainable": true, "dtype": "float32", "node_def": {"name": "Max_63", "op": "Max", "input": ["input_240", "Max_63/reduction_indices"], "attr": {"T": {"type": "DT_FLOAT"}, "keep_dims": {"b": true}, "Tidx": {"type": "DT_INT32"}}}, "constants": {"1": -1}}, "name": "tf_op_layer_Max_63", "inbound_nodes": [[["input_240", 0, 0, {}]]]}, {"class_name": "TensorFlowOpLayer", "config": {"name": "Mul_356", "trainable": true, "dtype": "float32", "node_def": {"name": "Mul_356", "op": "Mul", "input": ["Relu_55", "Max_63"], "attr": {"T": {"type": "DT_FLOAT"}}}, "constants": {}}, "name": "tf_op_layer_Mul_356", "inbound_nodes": [[["tf_op_layer_Relu_55", 0, 0, {}], ["tf_op_layer_Max_63", 0, 0, {}]]]}], "input_layers": [["latent_params", 0, 0], ["conditional_params", 0, 0], ["input_240", 0, 0]], "output_layers": [["tf_op_layer_Mul_356", 0, 0]]}}}
�"�
_tf_keras_input_layer�{"class_name": "InputLayer", "name": "latent_params", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 6]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 6]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "latent_params"}}
�
	variables
regularization_losses
trainable_variables
	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "RepeatVector", "name": "repeat_vector_59", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "repeat_vector_59", "trainable": true, "dtype": "float32", "n": 32}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 2, "max_ndim": null, "min_ndim": null, "axes": {}}}}
�"�
_tf_keras_input_layer�{"class_name": "InputLayer", "name": "conditional_params", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 32, 1]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 32, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "conditional_params"}}
�
 	variables
!regularization_losses
"trainable_variables
#	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "TensorFlowOpLayer", "name": "tf_op_layer_strided_slice_480", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "strided_slice_480", "trainable": true, "dtype": "float32", "node_def": {"name": "strided_slice_480", "op": "StridedSlice", "input": ["repeat_vector_59/Identity", "strided_slice_480/begin", "strided_slice_480/end", "strided_slice_480/strides"], "attr": {"end_mask": {"i": "0"}, "begin_mask": {"i": "0"}, "shrink_axis_mask": {"i": "0"}, "new_axis_mask": {"i": "0"}, "T": {"type": "DT_FLOAT"}, "ellipsis_mask": {"i": "1"}, "Index": {"type": "DT_INT32"}}}, "constants": {"1": [0, 0], "2": [0, 1], "3": [1, 1]}}}
�
$	variables
%regularization_losses
&trainable_variables
'	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "TensorFlowOpLayer", "name": "tf_op_layer_strided_slice_483", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "strided_slice_483", "trainable": true, "dtype": "float32", "node_def": {"name": "strided_slice_483", "op": "StridedSlice", "input": ["repeat_vector_59/Identity", "strided_slice_483/begin", "strided_slice_483/end", "strided_slice_483/strides"], "attr": {"new_axis_mask": {"i": "0"}, "begin_mask": {"i": "0"}, "end_mask": {"i": "2"}, "Index": {"type": "DT_INT32"}, "ellipsis_mask": {"i": "1"}, "shrink_axis_mask": {"i": "0"}, "T": {"type": "DT_FLOAT"}}}, "constants": {"1": [0, 3], "2": [0, 0], "3": [1, 1]}}}
�
(	variables
)regularization_losses
*trainable_variables
+	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "TensorFlowOpLayer", "name": "tf_op_layer_AddV2_118", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "AddV2_118", "trainable": true, "dtype": "float32", "node_def": {"name": "AddV2_118", "op": "AddV2", "input": ["conditional_params_62", "strided_slice_480"], "attr": {"T": {"type": "DT_FLOAT"}}}, "constants": {}}}
�
,	variables
-regularization_losses
.trainable_variables
/	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "Concatenate", "name": "concatenate_179", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "concatenate_179", "trainable": true, "dtype": "float32", "axis": -1}, "build_input_shape": [{"class_name": "TensorShape", "items": [null, 32, 3]}, {"class_name": "TensorShape", "items": [null, 32, 1]}]}
�
0	variables
1regularization_losses
2trainable_variables
3	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "TensorFlowOpLayer", "name": "tf_op_layer_strided_slice_482", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "strided_slice_482", "trainable": true, "dtype": "float32", "node_def": {"name": "strided_slice_482", "op": "StridedSlice", "input": ["repeat_vector_59/Identity", "strided_slice_482/begin", "strided_slice_482/end", "strided_slice_482/strides"], "attr": {"ellipsis_mask": {"i": "1"}, "shrink_axis_mask": {"i": "0"}, "T": {"type": "DT_FLOAT"}, "begin_mask": {"i": "0"}, "Index": {"type": "DT_INT32"}, "new_axis_mask": {"i": "0"}, "end_mask": {"i": "0"}}}, "constants": {"1": [0, 2], "2": [0, 3], "3": [1, 1]}}}
�

4kernel
5bias
6	variables
7regularization_losses
8trainable_variables
9	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "Dense", "name": "dense_476", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "dense_476", "trainable": true, "dtype": "float32", "units": 32, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 4}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 32, 4]}}
�4

:kernel
;	variables
<regularization_losses
=trainable_variables
>	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�2
_tf_keras_layer�2{"class_name": "Dense", "name": "color_law", "trainable": false, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "color_law", "trainable": false, "dtype": "float32", "units": 288, "activation": "linear", "use_bias": false, "kernel_initializer": {"class_name": "Constant", "config": {"value": [1.733986283286547, 1.7287693811029068, 1.7235902690277825, 1.7184478730039496, 1.713341136989258, 1.7082690226720982, 1.703230509249906, 1.698224593198338, 1.693250288043143, 1.688306624134713, 1.6833926484252757, 1.6785074242486995, 1.673650031102894, 1.6688195644347634, 1.664015135427689, 1.659235870791517, 1.6544809125550193, 1.6497494178608025, 1.6450405587626389, 1.6403535220251868, 1.635687508926083, 1.631041735060371, 1.626415430147253, 1.6218078378391225, 1.6172182155328652, 1.6126458341834013, 1.6080899781194373, 1.603549944861411, 1.5990250449416044, 1.5945146017263927, 1.5900179512406165, 1.5855344419940425, 1.5810634348099024, 1.5766043026554741, 1.572156430474695, 1.567719215022769, 1.5632920647027608, 1.5588743994041487, 1.554465650343312, 1.5500652599059337, 1.5456726814912982, 1.5412873793584616, 1.5369088284742718, 1.532536514363226, 1.52816993295913, 1.5238085904585568, 1.5194520031760688, 1.5150996974011943, 1.510751209257138, 1.506406084561195, 1.5020638786868683, 1.4977241564276482, 1.4933864918624562, 1.4890504682227241, 1.4847156777610842, 1.4803817216216668, 1.4760482097119774, 1.471714760576336, 1.4673810012708666, 1.4630465672400124, 1.458711102194566, 1.4543742579911925, 1.4500356945134296, 1.4456950795541552, 1.441352088699493, 1.4370064052141502, 1.4326577170773538, 1.4283052994820302, 1.42394758628978, 1.4195829394917132, 1.4152097553571679, 1.4108264639503714, 1.4064315286529432, 1.4020234456921643, 1.3976007436749458, 1.3931619831274442, 1.3887057560402456, 1.384230685419072, 1.379735424840937, 1.3752186580156902, 1.3706790983529, 1.3661154885340008, 1.3615266000896553, 1.3569112329822757, 1.3522682151936312, 1.3475964023175002, 1.342894677157307, 1.3381619493286763, 1.3333971548668704, 1.3285992558390292, 1.323767239961183, 1.318900120219965, 1.3139969344989837, 1.3090567452097988, 1.3040786389274424, 1.2990617260304427, 1.2940051403452952, 1.288908038795336, 1.2837696010539528, 1.2785890292021074, 1.2733655473901047, 1.2680984015035532, 1.2627868588334983, 1.257430207750652, 1.2520277573836838, 1.246579346471573, 1.2410871323811035, 1.2355538901996845, 1.2299823462385933, 1.2243751786311778, 1.2187350179713072, 1.2130644479443282, 1.2073660059506393, 1.201642183721952, 1.195895427930307, 1.1901281407899522, 1.1843426806521349, 1.1785413625929024, 1.1727264589939799, 1.166900200116804, 1.1610647746697969, 1.1552223303689295, 1.1493749744916821, 1.1435247744244514, 1.1376737582034773, 1.1318239150493759, 1.1259771958953362, 1.1201355139090525, 1.1143007450084679, 1.1084747283713863, 1.1026592669390394, 1.096856127913651, 1.0910670432500833, 1.0852937101416318, 1.0795377915000135, 1.0738009164296394, 1.0680846806962128, 1.0623906471897342, 1.056720346381954, 1.05107527677836, 1.0454569053647416, 1.0398666680483932, 1.034305970094022, 1.0287761865544245, 1.0232786626959696, 1.017814714418972, 1.0123856286729889, 1.0069926638671147, 1.0016370502753202, 0.9963199904368925, 0.9910426595520385, 0.9858062058726873, 0.9806116792336146, 0.975458939804804, 0.9703469032967426, 0.965274475843901, 0.9602405815868156, 0.9552441624369885, 0.9502841778445262, 0.9453596045684957, 0.9404694364499552, 0.935612684187638, 0.9307883751162611, 0.9259955529874249, 0.9212332777530796, 0.9165006253515322, 0.9117966874959529, 0.9071205714653741, 0.9024713998981359, 0.8978483105877624, 0.8932504562812408, 0.8886770044796674, 0.8841271372412491, 0.8796000509866208, 0.8750949563064638, 0.8706110777713957, 0.8661476537441003, 0.8617039361936851, 0.8572791905122311, 0.8528726953335185, 0.8484838197847225, 0.8441124032870609, 0.8397584556982287, 0.8354219857892379, 0.831103000996631, 0.8268015074441755, 0.822517509964287, 0.8182510121191809, 0.814002016221756, 0.8097705233562216, 0.8055565333984581, 0.8013600450361216, 0.7971810557885, 0.7930195620261065, 0.7888755589900335, 0.7847490408110609, 0.7806400005285148, 0.7765484301088936, 0.7724743204642488, 0.7684176614703389, 0.7643784419845474, 0.7603566498635708, 0.7563522719808821, 0.7523652942439681, 0.7483957016113454, 0.7444434781093592, 0.7405086068487651, 0.7365910700410948, 0.7326908490148105, 0.7288079242312535, 0.7249422753003828, 0.7210938809963111, 0.7172627192726369, 0.7134487672775773, 0.7096520013689066, 0.705872397128695, 0.7021099293778591, 0.6983645721905184, 0.6946362989081626, 0.6909250821536369, 0.6872308938449425, 0.6835537052088484, 0.6798934867943335, 0.6762502084858424, 0.6726238395163708, 0.6690143484803764, 0.6654217033465153, 0.661845871470212, 0.658286819606059, 0.6547445139200541, 0.6512189200016723, 0.6477100028757729, 0.6442177270143525, 0.6407420563481341, 0.6372829542780033, 0.6338403836862915, 0.6304143069479011, 0.6270046859412867, 0.6236114820592785, 0.6202346562197689, 0.6168741688762457, 0.6135299800281842, 0.6102020492312966, 0.6068903356076423, 0.6035947978555963, 0.6003153942596847, 0.5970520827002805, 0.5938048206631684, 0.5905735652489754, 0.5873582731824692, 0.5841589008217343, 0.5809754041672106, 0.5778077388706131, 0.5746558602437234, 0.5715197232670574, 0.5683992825984165, 0.5652944925813117, 0.5622053072532722, 0.5591316803540352, 0.5560735653336232, 0.5530309153603018, 0.5500036833284264, 0.5469918218661778, 0.5439952833431848, 0.5410140198780397, 0.5380479833457052, 0.5350971253848144, 0.5321613974048656, 0.5292407505933108, 0.5263351359225454, 0.523444504156795, 0.5205688058588979, 0.5177079913969939, 0.5148620109511113, 0.5120308145196603, 0.5092143519258254, 0.5064125728238699, 0.5036254267053438, 0.5008528629051977, 0.49809483060780846, 0.49535127885291513, 0.4926221565414631, 0.489907412441364, 0.48720699519316507, 0.48452085331563677, 0.4818489352112722, 0.47919118916554737, 0.4765475633769238]}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 1}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 32, 1]}}
�
?	variables
@regularization_losses
Atrainable_variables
B	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "TensorFlowOpLayer", "name": "tf_op_layer_strided_slice_481", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "strided_slice_481", "trainable": true, "dtype": "float32", "node_def": {"name": "strided_slice_481", "op": "StridedSlice", "input": ["repeat_vector_59/Identity", "strided_slice_481/begin", "strided_slice_481/end", "strided_slice_481/strides"], "attr": {"ellipsis_mask": {"i": "1"}, "T": {"type": "DT_FLOAT"}, "Index": {"type": "DT_INT32"}, "shrink_axis_mask": {"i": "0"}, "begin_mask": {"i": "0"}, "new_axis_mask": {"i": "0"}, "end_mask": {"i": "0"}}}, "constants": {"1": [0, 1], "2": [0, 2], "3": [1, 1]}}}
�

Ckernel
Dbias
E	variables
Fregularization_losses
Gtrainable_variables
H	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "Dense", "name": "dense_477", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "dense_477", "trainable": true, "dtype": "float32", "units": 128, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 32, 32]}}
�
I	variables
Jregularization_losses
Ktrainable_variables
L	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "TensorFlowOpLayer", "name": "tf_op_layer_AddV2_119", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "AddV2_119", "trainable": true, "dtype": "float32", "node_def": {"name": "AddV2_119", "op": "AddV2", "input": ["color_law_62/Identity", "strided_slice_481"], "attr": {"T": {"type": "DT_FLOAT"}}}, "constants": {}}}
�

Mkernel
Nbias
O	variables
Pregularization_losses
Qtrainable_variables
R	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "Dense", "name": "dense_478", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "dense_478", "trainable": true, "dtype": "float32", "units": 256, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 128}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 32, 128]}}
�
S	variables
Tregularization_losses
Utrainable_variables
V	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "TensorFlowOpLayer", "name": "tf_op_layer_Mul_354", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "Mul_354", "trainable": true, "dtype": "float32", "node_def": {"name": "Mul_354", "op": "Mul", "input": ["Mul_354/x", "AddV2_119"], "attr": {"T": {"type": "DT_FLOAT"}}}, "constants": {"0": -0.4000000059604645}}}
�

Wkernel
Xbias
Y	variables
Zregularization_losses
[trainable_variables
\	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "Dense", "name": "dense_479", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "dense_479", "trainable": true, "dtype": "float32", "units": 288, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 256}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 32, 256]}}
�
]	variables
^regularization_losses
_trainable_variables
`	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "TensorFlowOpLayer", "name": "tf_op_layer_Pow_59", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "Pow_59", "trainable": true, "dtype": "float32", "node_def": {"name": "Pow_59", "op": "Pow", "input": ["Pow_59/x", "Mul_354"], "attr": {"T": {"type": "DT_FLOAT"}}}, "constants": {"0": 10.0}}}
�
a	variables
bregularization_losses
ctrainable_variables
d	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "TensorFlowOpLayer", "name": "tf_op_layer_Mul_355", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "Mul_355", "trainable": true, "dtype": "float32", "node_def": {"name": "Mul_355", "op": "Mul", "input": ["dense_479/Identity", "Pow_59"], "attr": {"T": {"type": "DT_FLOAT"}}}, "constants": {}}}
�"�
_tf_keras_input_layer�{"class_name": "InputLayer", "name": "input_240", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 32, 288]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 32, 288]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_240"}}
�
e	variables
fregularization_losses
gtrainable_variables
h	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "TensorFlowOpLayer", "name": "tf_op_layer_Relu_55", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "Relu_55", "trainable": true, "dtype": "float32", "node_def": {"name": "Relu_55", "op": "Relu", "input": ["Mul_355"], "attr": {"T": {"type": "DT_FLOAT"}}}, "constants": {}}}
�
i	variables
jregularization_losses
ktrainable_variables
l	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "TensorFlowOpLayer", "name": "tf_op_layer_Max_63", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "Max_63", "trainable": true, "dtype": "float32", "node_def": {"name": "Max_63", "op": "Max", "input": ["input_240", "Max_63/reduction_indices"], "attr": {"T": {"type": "DT_FLOAT"}, "keep_dims": {"b": true}, "Tidx": {"type": "DT_INT32"}}}, "constants": {"1": -1}}}
�
m	variables
nregularization_losses
otrainable_variables
p	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "TensorFlowOpLayer", "name": "tf_op_layer_Mul_356", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "Mul_356", "trainable": true, "dtype": "float32", "node_def": {"name": "Mul_356", "op": "Mul", "input": ["Relu_55", "Max_63"], "attr": {"T": {"type": "DT_FLOAT"}}}, "constants": {}}}
_
40
51
:2
C3
D4
M5
N6
W7
X8"
trackable_list_wrapper
 "
trackable_list_wrapper
X
40
51
C2
D3
M4
N5
W6
X7"
trackable_list_wrapper
�
qmetrics
rlayer_regularization_losses
	variables
regularization_losses
trainable_variables
snon_trainable_variables
tlayer_metrics

ulayers
�__call__
�_default_save_signature
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
-
�serving_default"
signature_map
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
vmetrics
wlayer_regularization_losses
	variables
regularization_losses
trainable_variables
xnon_trainable_variables
ylayer_metrics

zlayers
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
{metrics
|layer_regularization_losses
 	variables
!regularization_losses
"trainable_variables
}non_trainable_variables
~layer_metrics

layers
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�metrics
 �layer_regularization_losses
$	variables
%regularization_losses
&trainable_variables
�non_trainable_variables
�layer_metrics
�layers
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�metrics
 �layer_regularization_losses
(	variables
)regularization_losses
*trainable_variables
�non_trainable_variables
�layer_metrics
�layers
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�metrics
 �layer_regularization_losses
,	variables
-regularization_losses
.trainable_variables
�non_trainable_variables
�layer_metrics
�layers
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�metrics
 �layer_regularization_losses
0	variables
1regularization_losses
2trainable_variables
�non_trainable_variables
�layer_metrics
�layers
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
":  2dense_476/kernel
: 2dense_476/bias
.
40
51"
trackable_list_wrapper
 "
trackable_list_wrapper
.
40
51"
trackable_list_wrapper
�
�metrics
 �layer_regularization_losses
6	variables
7regularization_losses
8trainable_variables
�non_trainable_variables
�layer_metrics
�layers
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
&:$	�2color_law_62/kernel
'
:0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�metrics
 �layer_regularization_losses
;	variables
<regularization_losses
=trainable_variables
�non_trainable_variables
�layer_metrics
�layers
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�metrics
 �layer_regularization_losses
?	variables
@regularization_losses
Atrainable_variables
�non_trainable_variables
�layer_metrics
�layers
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
#:!	 �2dense_477/kernel
:�2dense_477/bias
.
C0
D1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
C0
D1"
trackable_list_wrapper
�
�metrics
 �layer_regularization_losses
E	variables
Fregularization_losses
Gtrainable_variables
�non_trainable_variables
�layer_metrics
�layers
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�metrics
 �layer_regularization_losses
I	variables
Jregularization_losses
Ktrainable_variables
�non_trainable_variables
�layer_metrics
�layers
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
$:"
��2dense_478/kernel
:�2dense_478/bias
.
M0
N1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
M0
N1"
trackable_list_wrapper
�
�metrics
 �layer_regularization_losses
O	variables
Pregularization_losses
Qtrainable_variables
�non_trainable_variables
�layer_metrics
�layers
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�metrics
 �layer_regularization_losses
S	variables
Tregularization_losses
Utrainable_variables
�non_trainable_variables
�layer_metrics
�layers
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
$:"
��2dense_479/kernel
:�2dense_479/bias
.
W0
X1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
W0
X1"
trackable_list_wrapper
�
�metrics
 �layer_regularization_losses
Y	variables
Zregularization_losses
[trainable_variables
�non_trainable_variables
�layer_metrics
�layers
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�metrics
 �layer_regularization_losses
]	variables
^regularization_losses
_trainable_variables
�non_trainable_variables
�layer_metrics
�layers
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�metrics
 �layer_regularization_losses
a	variables
bregularization_losses
ctrainable_variables
�non_trainable_variables
�layer_metrics
�layers
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�metrics
 �layer_regularization_losses
e	variables
fregularization_losses
gtrainable_variables
�non_trainable_variables
�layer_metrics
�layers
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�metrics
 �layer_regularization_losses
i	variables
jregularization_losses
ktrainable_variables
�non_trainable_variables
�layer_metrics
�layers
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�metrics
 �layer_regularization_losses
m	variables
nregularization_losses
otrainable_variables
�non_trainable_variables
�layer_metrics
�layers
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
'
:0"
trackable_list_wrapper
 "
trackable_dict_wrapper
�
0
1
2
3
4
5
6
7
	8

9
10
11
12
13
14
15
16
17
18
19
20
21"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
'
:0"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
�2�
!__inference__wrapped_model_461507�
���
FullArgSpec
args� 
varargsjargs
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *���
���
'�$
latent_params���������
0�-
conditional_params��������� 
(�%
	input_240��������� �
�2�
E__inference_model_119_layer_call_and_return_conditional_losses_461994
E__inference_model_119_layer_call_and_return_conditional_losses_462327
E__inference_model_119_layer_call_and_return_conditional_losses_462496
E__inference_model_119_layer_call_and_return_conditional_losses_461951�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
*__inference_model_119_layer_call_fn_462546
*__inference_model_119_layer_call_fn_462521
*__inference_model_119_layer_call_fn_462131
*__inference_model_119_layer_call_fn_462063�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
L__inference_repeat_vector_59_layer_call_and_return_conditional_losses_461516�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *&�#
!�������������������
�2�
1__inference_repeat_vector_59_layer_call_fn_461522�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *&�#
!�������������������
�2�
Y__inference_tf_op_layer_strided_slice_480_layer_call_and_return_conditional_losses_462554�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
>__inference_tf_op_layer_strided_slice_480_layer_call_fn_462559�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
Y__inference_tf_op_layer_strided_slice_483_layer_call_and_return_conditional_losses_462567�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
>__inference_tf_op_layer_strided_slice_483_layer_call_fn_462572�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
Q__inference_tf_op_layer_AddV2_118_layer_call_and_return_conditional_losses_462578�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
6__inference_tf_op_layer_AddV2_118_layer_call_fn_462584�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
K__inference_concatenate_179_layer_call_and_return_conditional_losses_462591�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
0__inference_concatenate_179_layer_call_fn_462597�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
Y__inference_tf_op_layer_strided_slice_482_layer_call_and_return_conditional_losses_462605�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
>__inference_tf_op_layer_strided_slice_482_layer_call_fn_462610�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
E__inference_dense_476_layer_call_and_return_conditional_losses_462641�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
*__inference_dense_476_layer_call_fn_462650�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
E__inference_color_law_layer_call_and_return_conditional_losses_462677�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
*__inference_color_law_layer_call_fn_462684�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
Y__inference_tf_op_layer_strided_slice_481_layer_call_and_return_conditional_losses_462692�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
>__inference_tf_op_layer_strided_slice_481_layer_call_fn_462697�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
E__inference_dense_477_layer_call_and_return_conditional_losses_462728�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
*__inference_dense_477_layer_call_fn_462737�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
Q__inference_tf_op_layer_AddV2_119_layer_call_and_return_conditional_losses_462743�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
6__inference_tf_op_layer_AddV2_119_layer_call_fn_462749�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
E__inference_dense_478_layer_call_and_return_conditional_losses_462780�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
*__inference_dense_478_layer_call_fn_462789�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
O__inference_tf_op_layer_Mul_354_layer_call_and_return_conditional_losses_462795�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
4__inference_tf_op_layer_Mul_354_layer_call_fn_462800�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
E__inference_dense_479_layer_call_and_return_conditional_losses_462830�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
*__inference_dense_479_layer_call_fn_462839�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
N__inference_tf_op_layer_Pow_59_layer_call_and_return_conditional_losses_462845�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
3__inference_tf_op_layer_Pow_59_layer_call_fn_462850�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
O__inference_tf_op_layer_Mul_355_layer_call_and_return_conditional_losses_462856�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
4__inference_tf_op_layer_Mul_355_layer_call_fn_462862�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
O__inference_tf_op_layer_Relu_55_layer_call_and_return_conditional_losses_462867�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
4__inference_tf_op_layer_Relu_55_layer_call_fn_462872�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
N__inference_tf_op_layer_Max_63_layer_call_and_return_conditional_losses_462878�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
3__inference_tf_op_layer_Max_63_layer_call_fn_462883�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
O__inference_tf_op_layer_Mul_356_layer_call_and_return_conditional_losses_462889�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
4__inference_tf_op_layer_Mul_356_layer_call_fn_462895�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
VBT
$__inference_signature_wrapper_462158conditional_params	input_240latent_params�
!__inference__wrapped_model_461507�	:45CDMNWX���
���
���
'�$
latent_params���������
0�-
conditional_params��������� 
(�%
	input_240��������� �
� "N�K
I
tf_op_layer_Mul_3562�/
tf_op_layer_Mul_356��������� ��
E__inference_color_law_layer_call_and_return_conditional_losses_462677d:3�0
)�&
$�!
inputs��������� 
� "*�'
 �
0��������� �
� �
*__inference_color_law_layer_call_fn_462684W:3�0
)�&
$�!
inputs��������� 
� "���������� ��
K__inference_concatenate_179_layer_call_and_return_conditional_losses_462591�b�_
X�U
S�P
&�#
inputs/0��������� 
&�#
inputs/1��������� 
� ")�&
�
0��������� 
� �
0__inference_concatenate_179_layer_call_fn_462597�b�_
X�U
S�P
&�#
inputs/0��������� 
&�#
inputs/1��������� 
� "���������� �
E__inference_dense_476_layer_call_and_return_conditional_losses_462641d453�0
)�&
$�!
inputs��������� 
� ")�&
�
0���������  
� �
*__inference_dense_476_layer_call_fn_462650W453�0
)�&
$�!
inputs��������� 
� "����������  �
E__inference_dense_477_layer_call_and_return_conditional_losses_462728eCD3�0
)�&
$�!
inputs���������  
� "*�'
 �
0��������� �
� �
*__inference_dense_477_layer_call_fn_462737XCD3�0
)�&
$�!
inputs���������  
� "���������� ��
E__inference_dense_478_layer_call_and_return_conditional_losses_462780fMN4�1
*�'
%�"
inputs��������� �
� "*�'
 �
0��������� �
� �
*__inference_dense_478_layer_call_fn_462789YMN4�1
*�'
%�"
inputs��������� �
� "���������� ��
E__inference_dense_479_layer_call_and_return_conditional_losses_462830fWX4�1
*�'
%�"
inputs��������� �
� "*�'
 �
0��������� �
� �
*__inference_dense_479_layer_call_fn_462839YWX4�1
*�'
%�"
inputs��������� �
� "���������� ��
E__inference_model_119_layer_call_and_return_conditional_losses_461951�	:45CDMNWX���
���
���
'�$
latent_params���������
0�-
conditional_params��������� 
(�%
	input_240��������� �
p

 
� "*�'
 �
0��������� �
� �
E__inference_model_119_layer_call_and_return_conditional_losses_461994�	:45CDMNWX���
���
���
'�$
latent_params���������
0�-
conditional_params��������� 
(�%
	input_240��������� �
p 

 
� "*�'
 �
0��������� �
� �
E__inference_model_119_layer_call_and_return_conditional_losses_462327�	:45CDMNWX���
���
x�u
"�
inputs/0���������
&�#
inputs/1��������� 
'�$
inputs/2��������� �
p

 
� "*�'
 �
0��������� �
� �
E__inference_model_119_layer_call_and_return_conditional_losses_462496�	:45CDMNWX���
���
x�u
"�
inputs/0���������
&�#
inputs/1��������� 
'�$
inputs/2��������� �
p 

 
� "*�'
 �
0��������� �
� �
*__inference_model_119_layer_call_fn_462063�	:45CDMNWX���
���
���
'�$
latent_params���������
0�-
conditional_params��������� 
(�%
	input_240��������� �
p

 
� "���������� ��
*__inference_model_119_layer_call_fn_462131�	:45CDMNWX���
���
���
'�$
latent_params���������
0�-
conditional_params��������� 
(�%
	input_240��������� �
p 

 
� "���������� ��
*__inference_model_119_layer_call_fn_462521�	:45CDMNWX���
���
x�u
"�
inputs/0���������
&�#
inputs/1��������� 
'�$
inputs/2��������� �
p

 
� "���������� ��
*__inference_model_119_layer_call_fn_462546�	:45CDMNWX���
���
x�u
"�
inputs/0���������
&�#
inputs/1��������� 
'�$
inputs/2��������� �
p 

 
� "���������� ��
L__inference_repeat_vector_59_layer_call_and_return_conditional_losses_461516n8�5
.�+
)�&
inputs������������������
� "2�/
(�%
0��������� ���������
� �
1__inference_repeat_vector_59_layer_call_fn_461522a8�5
.�+
)�&
inputs������������������
� "%�"��������� ����������
$__inference_signature_wrapper_462158�	:45CDMNWX���
� 
���
F
conditional_params0�-
conditional_params��������� 
5
	input_240(�%
	input_240��������� �
8
latent_params'�$
latent_params���������"N�K
I
tf_op_layer_Mul_3562�/
tf_op_layer_Mul_356��������� ��
Q__inference_tf_op_layer_AddV2_118_layer_call_and_return_conditional_losses_462578�b�_
X�U
S�P
&�#
inputs/0��������� 
&�#
inputs/1��������� 
� ")�&
�
0��������� 
� �
6__inference_tf_op_layer_AddV2_118_layer_call_fn_462584�b�_
X�U
S�P
&�#
inputs/0��������� 
&�#
inputs/1��������� 
� "���������� �
Q__inference_tf_op_layer_AddV2_119_layer_call_and_return_conditional_losses_462743�c�`
Y�V
T�Q
'�$
inputs/0��������� �
&�#
inputs/1��������� 
� "*�'
 �
0��������� �
� �
6__inference_tf_op_layer_AddV2_119_layer_call_fn_462749�c�`
Y�V
T�Q
'�$
inputs/0��������� �
&�#
inputs/1��������� 
� "���������� ��
N__inference_tf_op_layer_Max_63_layer_call_and_return_conditional_losses_462878a4�1
*�'
%�"
inputs��������� �
� ")�&
�
0��������� 
� �
3__inference_tf_op_layer_Max_63_layer_call_fn_462883T4�1
*�'
%�"
inputs��������� �
� "���������� �
O__inference_tf_op_layer_Mul_354_layer_call_and_return_conditional_losses_462795b4�1
*�'
%�"
inputs��������� �
� "*�'
 �
0��������� �
� �
4__inference_tf_op_layer_Mul_354_layer_call_fn_462800U4�1
*�'
%�"
inputs��������� �
� "���������� ��
O__inference_tf_op_layer_Mul_355_layer_call_and_return_conditional_losses_462856�d�a
Z�W
U�R
'�$
inputs/0��������� �
'�$
inputs/1��������� �
� "*�'
 �
0��������� �
� �
4__inference_tf_op_layer_Mul_355_layer_call_fn_462862�d�a
Z�W
U�R
'�$
inputs/0��������� �
'�$
inputs/1��������� �
� "���������� ��
O__inference_tf_op_layer_Mul_356_layer_call_and_return_conditional_losses_462889�c�`
Y�V
T�Q
'�$
inputs/0��������� �
&�#
inputs/1��������� 
� "*�'
 �
0��������� �
� �
4__inference_tf_op_layer_Mul_356_layer_call_fn_462895�c�`
Y�V
T�Q
'�$
inputs/0��������� �
&�#
inputs/1��������� 
� "���������� ��
N__inference_tf_op_layer_Pow_59_layer_call_and_return_conditional_losses_462845b4�1
*�'
%�"
inputs��������� �
� "*�'
 �
0��������� �
� �
3__inference_tf_op_layer_Pow_59_layer_call_fn_462850U4�1
*�'
%�"
inputs��������� �
� "���������� ��
O__inference_tf_op_layer_Relu_55_layer_call_and_return_conditional_losses_462867b4�1
*�'
%�"
inputs��������� �
� "*�'
 �
0��������� �
� �
4__inference_tf_op_layer_Relu_55_layer_call_fn_462872U4�1
*�'
%�"
inputs��������� �
� "���������� ��
Y__inference_tf_op_layer_strided_slice_480_layer_call_and_return_conditional_losses_462554`3�0
)�&
$�!
inputs��������� 
� ")�&
�
0��������� 
� �
>__inference_tf_op_layer_strided_slice_480_layer_call_fn_462559S3�0
)�&
$�!
inputs��������� 
� "���������� �
Y__inference_tf_op_layer_strided_slice_481_layer_call_and_return_conditional_losses_462692`3�0
)�&
$�!
inputs��������� 
� ")�&
�
0��������� 
� �
>__inference_tf_op_layer_strided_slice_481_layer_call_fn_462697S3�0
)�&
$�!
inputs��������� 
� "���������� �
Y__inference_tf_op_layer_strided_slice_482_layer_call_and_return_conditional_losses_462605`3�0
)�&
$�!
inputs��������� 
� ")�&
�
0��������� 
� �
>__inference_tf_op_layer_strided_slice_482_layer_call_fn_462610S3�0
)�&
$�!
inputs��������� 
� "���������� �
Y__inference_tf_op_layer_strided_slice_483_layer_call_and_return_conditional_losses_462567`3�0
)�&
$�!
inputs��������� 
� ")�&
�
0��������� 
� �
>__inference_tf_op_layer_strided_slice_483_layer_call_fn_462572S3�0
)�&
$�!
inputs��������� 
� "���������� 