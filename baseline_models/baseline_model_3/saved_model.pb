กป
โ
D
AddV2
x"T
y"T
z"T"
Ttype:
2	
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( 
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
8
Const
output"dtype"
valuetensor"
dtypetype

Conv2D

input"T
filter"T
output"T"
Ttype:	
2"
strides	list(int)"
use_cudnn_on_gpubool(",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 "-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

๛
FusedBatchNormV3
x"T

scale"U
offset"U	
mean"U
variance"U
y"T

batch_mean"U
batch_variance"U
reserve_space_1"U
reserve_space_2"U
reserve_space_3"U"
Ttype:
2"
Utype:
2"
epsilonfloat%ทั8"&
exponential_avg_factorfloat%  ?";
data_formatstringNHWC:
NHWCNCHWNDHWCNCDHW"
is_trainingbool(
.
Identity

input"T
output"T"	
Ttype
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	

MaxPool

input"T
output"T"
Ttype0:
2	"
ksize	list(int)(0"
strides	list(int)(0",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 ":
data_formatstringNHWC:
NHWCNCHWNCHW_VECT_C
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(
?
Mul
x"T
y"T
z"T"
Ttype:
2	

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype
E
Relu
features"T
activations"T"
Ttype:
2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
?
Select
	condition

t"T
e"T
output"T"	
Ttype
H
ShardedFilename
basename	
shard

num_shards
filename
9
Softmax
logits"T
softmax"T"
Ttype:
2
ม
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
executor_typestring จ
@
StaticRegexFullMatch	
input

output
"
patternstring
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 

VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 "serve*2.9.22v2.9.1-132-g18960c44ad38๓ค
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
b
count_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_1
[
count_1/Read/ReadVariableOpReadVariableOpcount_1*
_output_shapes
: *
dtype0
b
total_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_1
[
total_1/Read/ReadVariableOpReadVariableOptotal_1*
_output_shapes
: *
dtype0
p
dense_8/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_8/bias
i
 dense_8/bias/Read/ReadVariableOpReadVariableOpdense_8/bias*
_output_shapes
:*
dtype0
y
dense_8/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*
shared_namedense_8/kernel
r
"dense_8/kernel/Read/ReadVariableOpReadVariableOpdense_8/kernel*
_output_shapes
:	*
dtype0
q
dense_7/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_7/bias
j
 dense_7/bias/Read/ReadVariableOpReadVariableOpdense_7/bias*
_output_shapes	
:*
dtype0
z
dense_7/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*
shared_namedense_7/kernel
s
"dense_7/kernel/Read/ReadVariableOpReadVariableOpdense_7/kernel* 
_output_shapes
:
*
dtype0
q
dense_6/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_6/bias
j
 dense_6/bias/Read/ReadVariableOpReadVariableOpdense_6/bias*
_output_shapes	
:*
dtype0
z
dense_6/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*
shared_namedense_6/kernel
s
"dense_6/kernel/Read/ReadVariableOpReadVariableOpdense_6/kernel* 
_output_shapes
:
*
dtype0
ฅ
&batch_normalization_14/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*7
shared_name(&batch_normalization_14/moving_variance

:batch_normalization_14/moving_variance/Read/ReadVariableOpReadVariableOp&batch_normalization_14/moving_variance*
_output_shapes	
:*
dtype0

"batch_normalization_14/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"batch_normalization_14/moving_mean

6batch_normalization_14/moving_mean/Read/ReadVariableOpReadVariableOp"batch_normalization_14/moving_mean*
_output_shapes	
:*
dtype0

batch_normalization_14/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*,
shared_namebatch_normalization_14/beta

/batch_normalization_14/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_14/beta*
_output_shapes	
:*
dtype0

batch_normalization_14/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*-
shared_namebatch_normalization_14/gamma

0batch_normalization_14/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_14/gamma*
_output_shapes	
:*
dtype0
u
conv2d_14/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv2d_14/bias
n
"conv2d_14/bias/Read/ReadVariableOpReadVariableOpconv2d_14/bias*
_output_shapes	
:*
dtype0

conv2d_14/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*!
shared_nameconv2d_14/kernel

$conv2d_14/kernel/Read/ReadVariableOpReadVariableOpconv2d_14/kernel*(
_output_shapes
:*
dtype0
ฅ
&batch_normalization_13/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*7
shared_name(&batch_normalization_13/moving_variance

:batch_normalization_13/moving_variance/Read/ReadVariableOpReadVariableOp&batch_normalization_13/moving_variance*
_output_shapes	
:*
dtype0

"batch_normalization_13/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"batch_normalization_13/moving_mean

6batch_normalization_13/moving_mean/Read/ReadVariableOpReadVariableOp"batch_normalization_13/moving_mean*
_output_shapes	
:*
dtype0

batch_normalization_13/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*,
shared_namebatch_normalization_13/beta

/batch_normalization_13/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_13/beta*
_output_shapes	
:*
dtype0

batch_normalization_13/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*-
shared_namebatch_normalization_13/gamma

0batch_normalization_13/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_13/gamma*
_output_shapes	
:*
dtype0
u
conv2d_13/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv2d_13/bias
n
"conv2d_13/bias/Read/ReadVariableOpReadVariableOpconv2d_13/bias*
_output_shapes	
:*
dtype0

conv2d_13/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*!
shared_nameconv2d_13/kernel

$conv2d_13/kernel/Read/ReadVariableOpReadVariableOpconv2d_13/kernel*(
_output_shapes
:*
dtype0
ฅ
&batch_normalization_12/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*7
shared_name(&batch_normalization_12/moving_variance

:batch_normalization_12/moving_variance/Read/ReadVariableOpReadVariableOp&batch_normalization_12/moving_variance*
_output_shapes	
:*
dtype0

"batch_normalization_12/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"batch_normalization_12/moving_mean

6batch_normalization_12/moving_mean/Read/ReadVariableOpReadVariableOp"batch_normalization_12/moving_mean*
_output_shapes	
:*
dtype0

batch_normalization_12/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*,
shared_namebatch_normalization_12/beta

/batch_normalization_12/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_12/beta*
_output_shapes	
:*
dtype0

batch_normalization_12/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*-
shared_namebatch_normalization_12/gamma

0batch_normalization_12/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_12/gamma*
_output_shapes	
:*
dtype0
u
conv2d_12/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv2d_12/bias
n
"conv2d_12/bias/Read/ReadVariableOpReadVariableOpconv2d_12/bias*
_output_shapes	
:*
dtype0

conv2d_12/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*!
shared_nameconv2d_12/kernel

$conv2d_12/kernel/Read/ReadVariableOpReadVariableOpconv2d_12/kernel*(
_output_shapes
:*
dtype0
ฅ
&batch_normalization_11/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*7
shared_name(&batch_normalization_11/moving_variance

:batch_normalization_11/moving_variance/Read/ReadVariableOpReadVariableOp&batch_normalization_11/moving_variance*
_output_shapes	
:*
dtype0

"batch_normalization_11/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"batch_normalization_11/moving_mean

6batch_normalization_11/moving_mean/Read/ReadVariableOpReadVariableOp"batch_normalization_11/moving_mean*
_output_shapes	
:*
dtype0

batch_normalization_11/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*,
shared_namebatch_normalization_11/beta

/batch_normalization_11/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_11/beta*
_output_shapes	
:*
dtype0

batch_normalization_11/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*-
shared_namebatch_normalization_11/gamma

0batch_normalization_11/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_11/gamma*
_output_shapes	
:*
dtype0
u
conv2d_11/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv2d_11/bias
n
"conv2d_11/bias/Read/ReadVariableOpReadVariableOpconv2d_11/bias*
_output_shapes	
:*
dtype0

conv2d_11/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:`*!
shared_nameconv2d_11/kernel
~
$conv2d_11/kernel/Read/ReadVariableOpReadVariableOpconv2d_11/kernel*'
_output_shapes
:`*
dtype0
ค
&batch_normalization_10/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:`*7
shared_name(&batch_normalization_10/moving_variance

:batch_normalization_10/moving_variance/Read/ReadVariableOpReadVariableOp&batch_normalization_10/moving_variance*
_output_shapes
:`*
dtype0

"batch_normalization_10/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:`*3
shared_name$"batch_normalization_10/moving_mean

6batch_normalization_10/moving_mean/Read/ReadVariableOpReadVariableOp"batch_normalization_10/moving_mean*
_output_shapes
:`*
dtype0

batch_normalization_10/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:`*,
shared_namebatch_normalization_10/beta

/batch_normalization_10/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_10/beta*
_output_shapes
:`*
dtype0

batch_normalization_10/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:`*-
shared_namebatch_normalization_10/gamma

0batch_normalization_10/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_10/gamma*
_output_shapes
:`*
dtype0
t
conv2d_10/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:`*
shared_nameconv2d_10/bias
m
"conv2d_10/bias/Read/ReadVariableOpReadVariableOpconv2d_10/bias*
_output_shapes
:`*
dtype0

conv2d_10/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:`*!
shared_nameconv2d_10/kernel
}
$conv2d_10/kernel/Read/ReadVariableOpReadVariableOpconv2d_10/kernel*&
_output_shapes
:`*
dtype0

NoOpNoOp
ฒ
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*์
valueแB? Bี
ิ
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer-3
layer_with_weights-2
layer-4
layer_with_weights-3
layer-5
layer-6
layer_with_weights-4
layer-7
	layer_with_weights-5
	layer-8

layer_with_weights-6

layer-9
layer_with_weights-7
layer-10
layer_with_weights-8
layer-11
layer_with_weights-9
layer-12
layer-13
layer-14
layer_with_weights-10
layer-15
layer-16
layer_with_weights-11
layer-17
layer-18
layer_with_weights-12
layer-19
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature
	optimizer

signatures
#_self_saveable_object_factories*
ณ
	variables
 trainable_variables
!regularization_losses
"	keras_api
#__call__
*$&call_and_return_all_conditional_losses
#%_self_saveable_object_factories* 
ํ
&	variables
'trainable_variables
(regularization_losses
)	keras_api
*__call__
*+&call_and_return_all_conditional_losses

,kernel
-bias
#._self_saveable_object_factories
 /_jit_compiled_convolution_op*
๚
0	variables
1trainable_variables
2regularization_losses
3	keras_api
4__call__
*5&call_and_return_all_conditional_losses
6axis
	7gamma
8beta
9moving_mean
:moving_variance
#;_self_saveable_object_factories*
ณ
<	variables
=trainable_variables
>regularization_losses
?	keras_api
@__call__
*A&call_and_return_all_conditional_losses
#B_self_saveable_object_factories* 
ํ
C	variables
Dtrainable_variables
Eregularization_losses
F	keras_api
G__call__
*H&call_and_return_all_conditional_losses

Ikernel
Jbias
#K_self_saveable_object_factories
 L_jit_compiled_convolution_op*
๚
M	variables
Ntrainable_variables
Oregularization_losses
P	keras_api
Q__call__
*R&call_and_return_all_conditional_losses
Saxis
	Tgamma
Ubeta
Vmoving_mean
Wmoving_variance
#X_self_saveable_object_factories*
ณ
Y	variables
Ztrainable_variables
[regularization_losses
\	keras_api
]__call__
*^&call_and_return_all_conditional_losses
#__self_saveable_object_factories* 
ํ
`	variables
atrainable_variables
bregularization_losses
c	keras_api
d__call__
*e&call_and_return_all_conditional_losses

fkernel
gbias
#h_self_saveable_object_factories
 i_jit_compiled_convolution_op*
๚
j	variables
ktrainable_variables
lregularization_losses
m	keras_api
n__call__
*o&call_and_return_all_conditional_losses
paxis
	qgamma
rbeta
smoving_mean
tmoving_variance
#u_self_saveable_object_factories*
ํ
v	variables
wtrainable_variables
xregularization_losses
y	keras_api
z__call__
*{&call_and_return_all_conditional_losses

|kernel
}bias
#~_self_saveable_object_factories
 _jit_compiled_convolution_op*

	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses
	axis

gamma
	beta
moving_mean
moving_variance
$_self_saveable_object_factories*
๗
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses
kernel
	bias
$_self_saveable_object_factories
!_jit_compiled_convolution_op*

	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses
	axis

gamma
	beta
moving_mean
?moving_variance
$ก_self_saveable_object_factories*
บ
ข	variables
ฃtrainable_variables
คregularization_losses
ฅ	keras_api
ฆ__call__
+ง&call_and_return_all_conditional_losses
$จ_self_saveable_object_factories* 
บ
ฉ	variables
ชtrainable_variables
ซregularization_losses
ฌ	keras_api
ญ__call__
+ฎ&call_and_return_all_conditional_losses
$ฏ_self_saveable_object_factories* 
ิ
ฐ	variables
ฑtrainable_variables
ฒregularization_losses
ณ	keras_api
ด__call__
+ต&call_and_return_all_conditional_losses
ถkernel
	ทbias
$ธ_self_saveable_object_factories*
า
น	variables
บtrainable_variables
ปregularization_losses
ผ	keras_api
ฝ__call__
+พ&call_and_return_all_conditional_losses
ฟ_random_generator
$ภ_self_saveable_object_factories* 
ิ
ม	variables
ยtrainable_variables
รregularization_losses
ฤ	keras_api
ล__call__
+ฦ&call_and_return_all_conditional_losses
วkernel
	ศbias
$ษ_self_saveable_object_factories*
า
ส	variables
หtrainable_variables
ฬregularization_losses
อ	keras_api
ฮ__call__
+ฯ&call_and_return_all_conditional_losses
ะ_random_generator
$ั_self_saveable_object_factories* 
ิ
า	variables
ำtrainable_variables
ิregularization_losses
ี	keras_api
ึ__call__
+ื&call_and_return_all_conditional_losses
ุkernel
	ูbias
$ฺ_self_saveable_object_factories*
ช
,0
-1
72
83
94
:5
I6
J7
T8
U9
V10
W11
f12
g13
q14
r15
s16
t17
|18
}19
20
21
22
23
24
25
26
27
28
?29
ถ30
ท31
ว32
ศ33
ุ34
ู35*
ึ
,0
-1
72
83
I4
J5
T6
U7
f8
g9
q10
r11
|12
}13
14
15
16
17
18
19
ถ20
ท21
ว22
ศ23
ุ24
ู25*
* 
ต
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
฿layer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
:
เtrace_0
แtrace_1
โtrace_2
ใtrace_3* 
:
ไtrace_0
ๅtrace_1
ๆtrace_2
็trace_3* 
* 
* 

่serving_default* 
* 
* 
* 
* 

้non_trainable_variables
๊layers
๋metrics
 ์layer_regularization_losses
ํlayer_metrics
	variables
 trainable_variables
!regularization_losses
#__call__
*$&call_and_return_all_conditional_losses
&$"call_and_return_conditional_losses* 

๎trace_0* 

๏trace_0* 
* 

,0
-1*

,0
-1*
* 

๐non_trainable_variables
๑layers
๒metrics
 ๓layer_regularization_losses
๔layer_metrics
&	variables
'trainable_variables
(regularization_losses
*__call__
*+&call_and_return_all_conditional_losses
&+"call_and_return_conditional_losses*

๕trace_0* 

๖trace_0* 
`Z
VARIABLE_VALUEconv2d_10/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUEconv2d_10/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
 
70
81
92
:3*

70
81*
* 

๗non_trainable_variables
๘layers
๙metrics
 ๚layer_regularization_losses
๛layer_metrics
0	variables
1trainable_variables
2regularization_losses
4__call__
*5&call_and_return_all_conditional_losses
&5"call_and_return_conditional_losses*

?trace_0
?trace_1* 

?trace_0
?trace_1* 
* 
ke
VARIABLE_VALUEbatch_normalization_10/gamma5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUE*
ic
VARIABLE_VALUEbatch_normalization_10/beta4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUE*
wq
VARIABLE_VALUE"batch_normalization_10/moving_mean;layer_with_weights-1/moving_mean/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUE&batch_normalization_10/moving_variance?layer_with_weights-1/moving_variance/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
<	variables
=trainable_variables
>regularization_losses
@__call__
*A&call_and_return_all_conditional_losses
&A"call_and_return_conditional_losses* 

trace_0* 

trace_0* 
* 

I0
J1*

I0
J1*
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
C	variables
Dtrainable_variables
Eregularization_losses
G__call__
*H&call_and_return_all_conditional_losses
&H"call_and_return_conditional_losses*

trace_0* 

trace_0* 
`Z
VARIABLE_VALUEconv2d_11/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUEconv2d_11/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
 
T0
U1
V2
W3*

T0
U1*
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
M	variables
Ntrainable_variables
Oregularization_losses
Q__call__
*R&call_and_return_all_conditional_losses
&R"call_and_return_conditional_losses*

trace_0
trace_1* 

trace_0
trace_1* 
* 
ke
VARIABLE_VALUEbatch_normalization_11/gamma5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUE*
ic
VARIABLE_VALUEbatch_normalization_11/beta4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUE*
wq
VARIABLE_VALUE"batch_normalization_11/moving_mean;layer_with_weights-3/moving_mean/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUE&batch_normalization_11/moving_variance?layer_with_weights-3/moving_variance/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
Y	variables
Ztrainable_variables
[regularization_losses
]__call__
*^&call_and_return_all_conditional_losses
&^"call_and_return_conditional_losses* 

trace_0* 

trace_0* 
* 

f0
g1*

f0
g1*
* 

non_trainable_variables
layers
?metrics
 กlayer_regularization_losses
ขlayer_metrics
`	variables
atrainable_variables
bregularization_losses
d__call__
*e&call_and_return_all_conditional_losses
&e"call_and_return_conditional_losses*

ฃtrace_0* 

คtrace_0* 
`Z
VARIABLE_VALUEconv2d_12/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUEconv2d_12/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
 
q0
r1
s2
t3*

q0
r1*
* 

ฅnon_trainable_variables
ฆlayers
งmetrics
 จlayer_regularization_losses
ฉlayer_metrics
j	variables
ktrainable_variables
lregularization_losses
n__call__
*o&call_and_return_all_conditional_losses
&o"call_and_return_conditional_losses*

ชtrace_0
ซtrace_1* 

ฌtrace_0
ญtrace_1* 
* 
ke
VARIABLE_VALUEbatch_normalization_12/gamma5layer_with_weights-5/gamma/.ATTRIBUTES/VARIABLE_VALUE*
ic
VARIABLE_VALUEbatch_normalization_12/beta4layer_with_weights-5/beta/.ATTRIBUTES/VARIABLE_VALUE*
wq
VARIABLE_VALUE"batch_normalization_12/moving_mean;layer_with_weights-5/moving_mean/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUE&batch_normalization_12/moving_variance?layer_with_weights-5/moving_variance/.ATTRIBUTES/VARIABLE_VALUE*
* 

|0
}1*

|0
}1*
* 

ฎnon_trainable_variables
ฏlayers
ฐmetrics
 ฑlayer_regularization_losses
ฒlayer_metrics
v	variables
wtrainable_variables
xregularization_losses
z__call__
*{&call_and_return_all_conditional_losses
&{"call_and_return_conditional_losses*

ณtrace_0* 

ดtrace_0* 
`Z
VARIABLE_VALUEconv2d_13/kernel6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUEconv2d_13/bias4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
$
0
1
2
3*

0
1*
* 

ตnon_trainable_variables
ถlayers
ทmetrics
 ธlayer_regularization_losses
นlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses*

บtrace_0
ปtrace_1* 

ผtrace_0
ฝtrace_1* 
* 
ke
VARIABLE_VALUEbatch_normalization_13/gamma5layer_with_weights-7/gamma/.ATTRIBUTES/VARIABLE_VALUE*
ic
VARIABLE_VALUEbatch_normalization_13/beta4layer_with_weights-7/beta/.ATTRIBUTES/VARIABLE_VALUE*
wq
VARIABLE_VALUE"batch_normalization_13/moving_mean;layer_with_weights-7/moving_mean/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUE&batch_normalization_13/moving_variance?layer_with_weights-7/moving_variance/.ATTRIBUTES/VARIABLE_VALUE*
* 

0
1*

0
1*
* 

พnon_trainable_variables
ฟlayers
ภmetrics
 มlayer_regularization_losses
ยlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses*

รtrace_0* 

ฤtrace_0* 
`Z
VARIABLE_VALUEconv2d_14/kernel6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUEconv2d_14/bias4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
$
0
1
2
?3*

0
1*
* 

ลnon_trainable_variables
ฦlayers
วmetrics
 ศlayer_regularization_losses
ษlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses*

สtrace_0
หtrace_1* 

ฬtrace_0
อtrace_1* 
* 
ke
VARIABLE_VALUEbatch_normalization_14/gamma5layer_with_weights-9/gamma/.ATTRIBUTES/VARIABLE_VALUE*
ic
VARIABLE_VALUEbatch_normalization_14/beta4layer_with_weights-9/beta/.ATTRIBUTES/VARIABLE_VALUE*
wq
VARIABLE_VALUE"batch_normalization_14/moving_mean;layer_with_weights-9/moving_mean/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUE&batch_normalization_14/moving_variance?layer_with_weights-9/moving_variance/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 

ฮnon_trainable_variables
ฯlayers
ะmetrics
 ัlayer_regularization_losses
าlayer_metrics
ข	variables
ฃtrainable_variables
คregularization_losses
ฆ__call__
+ง&call_and_return_all_conditional_losses
'ง"call_and_return_conditional_losses* 

ำtrace_0* 

ิtrace_0* 
* 
* 
* 
* 

ีnon_trainable_variables
ึlayers
ืmetrics
 ุlayer_regularization_losses
ูlayer_metrics
ฉ	variables
ชtrainable_variables
ซregularization_losses
ญ__call__
+ฎ&call_and_return_all_conditional_losses
'ฎ"call_and_return_conditional_losses* 

ฺtrace_0* 

?trace_0* 
* 

ถ0
ท1*

ถ0
ท1*
* 

?non_trainable_variables
?layers
?metrics
 ฿layer_regularization_losses
เlayer_metrics
ฐ	variables
ฑtrainable_variables
ฒregularization_losses
ด__call__
+ต&call_and_return_all_conditional_losses
'ต"call_and_return_conditional_losses*

แtrace_0* 

โtrace_0* 
_Y
VARIABLE_VALUEdense_6/kernel7layer_with_weights-10/kernel/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEdense_6/bias5layer_with_weights-10/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 

ใnon_trainable_variables
ไlayers
ๅmetrics
 ๆlayer_regularization_losses
็layer_metrics
น	variables
บtrainable_variables
ปregularization_losses
ฝ__call__
+พ&call_and_return_all_conditional_losses
'พ"call_and_return_conditional_losses* 

่trace_0
้trace_1* 

๊trace_0
๋trace_1* 
* 
* 

ว0
ศ1*

ว0
ศ1*
* 

์non_trainable_variables
ํlayers
๎metrics
 ๏layer_regularization_losses
๐layer_metrics
ม	variables
ยtrainable_variables
รregularization_losses
ล__call__
+ฦ&call_and_return_all_conditional_losses
'ฦ"call_and_return_conditional_losses*

๑trace_0* 

๒trace_0* 
_Y
VARIABLE_VALUEdense_7/kernel7layer_with_weights-11/kernel/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEdense_7/bias5layer_with_weights-11/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 

๓non_trainable_variables
๔layers
๕metrics
 ๖layer_regularization_losses
๗layer_metrics
ส	variables
หtrainable_variables
ฬregularization_losses
ฮ__call__
+ฯ&call_and_return_all_conditional_losses
'ฯ"call_and_return_conditional_losses* 

๘trace_0
๙trace_1* 

๚trace_0
๛trace_1* 
* 
* 

ุ0
ู1*

ุ0
ู1*
* 

?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
layer_metrics
า	variables
ำtrainable_variables
ิregularization_losses
ึ__call__
+ื&call_and_return_all_conditional_losses
'ื"call_and_return_conditional_losses*

trace_0* 

trace_0* 
_Y
VARIABLE_VALUEdense_8/kernel7layer_with_weights-12/kernel/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEdense_8/bias5layer_with_weights-12/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
N
90
:1
V2
W3
s4
t5
6
7
8
?9*

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
19*

0
1*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

90
:1*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

V0
W1*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

s0
t1*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

0
1*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

0
?1*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
<
	variables
	keras_api

total

count*
M
	variables
	keras_api

total

count

_fn_kwargs*

0
1*

	variables*
UO
VARIABLE_VALUEtotal_14keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEcount_14keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE*

0
1*

	variables*
SM
VARIABLE_VALUEtotal4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEcount4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE*
* 

!serving_default_rescaling_2_inputPlaceholder*1
_output_shapes
:?????????เ*
dtype0*&
shape:?????????เ
ช

StatefulPartitionedCallStatefulPartitionedCall!serving_default_rescaling_2_inputconv2d_10/kernelconv2d_10/biasbatch_normalization_10/gammabatch_normalization_10/beta"batch_normalization_10/moving_mean&batch_normalization_10/moving_varianceconv2d_11/kernelconv2d_11/biasbatch_normalization_11/gammabatch_normalization_11/beta"batch_normalization_11/moving_mean&batch_normalization_11/moving_varianceconv2d_12/kernelconv2d_12/biasbatch_normalization_12/gammabatch_normalization_12/beta"batch_normalization_12/moving_mean&batch_normalization_12/moving_varianceconv2d_13/kernelconv2d_13/biasbatch_normalization_13/gammabatch_normalization_13/beta"batch_normalization_13/moving_mean&batch_normalization_13/moving_varianceconv2d_14/kernelconv2d_14/biasbatch_normalization_14/gammabatch_normalization_14/beta"batch_normalization_14/moving_mean&batch_normalization_14/moving_variancedense_6/kerneldense_6/biasdense_7/kerneldense_7/biasdense_8/kerneldense_8/bias*0
Tin)
'2%*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*F
_read_only_resource_inputs(
&$	
 !"#$*-
config_proto

CPU

GPU 2J 8 *,
f'R%
#__inference_signature_wrapper_92515
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
ฉ
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename$conv2d_10/kernel/Read/ReadVariableOp"conv2d_10/bias/Read/ReadVariableOp0batch_normalization_10/gamma/Read/ReadVariableOp/batch_normalization_10/beta/Read/ReadVariableOp6batch_normalization_10/moving_mean/Read/ReadVariableOp:batch_normalization_10/moving_variance/Read/ReadVariableOp$conv2d_11/kernel/Read/ReadVariableOp"conv2d_11/bias/Read/ReadVariableOp0batch_normalization_11/gamma/Read/ReadVariableOp/batch_normalization_11/beta/Read/ReadVariableOp6batch_normalization_11/moving_mean/Read/ReadVariableOp:batch_normalization_11/moving_variance/Read/ReadVariableOp$conv2d_12/kernel/Read/ReadVariableOp"conv2d_12/bias/Read/ReadVariableOp0batch_normalization_12/gamma/Read/ReadVariableOp/batch_normalization_12/beta/Read/ReadVariableOp6batch_normalization_12/moving_mean/Read/ReadVariableOp:batch_normalization_12/moving_variance/Read/ReadVariableOp$conv2d_13/kernel/Read/ReadVariableOp"conv2d_13/bias/Read/ReadVariableOp0batch_normalization_13/gamma/Read/ReadVariableOp/batch_normalization_13/beta/Read/ReadVariableOp6batch_normalization_13/moving_mean/Read/ReadVariableOp:batch_normalization_13/moving_variance/Read/ReadVariableOp$conv2d_14/kernel/Read/ReadVariableOp"conv2d_14/bias/Read/ReadVariableOp0batch_normalization_14/gamma/Read/ReadVariableOp/batch_normalization_14/beta/Read/ReadVariableOp6batch_normalization_14/moving_mean/Read/ReadVariableOp:batch_normalization_14/moving_variance/Read/ReadVariableOp"dense_6/kernel/Read/ReadVariableOp dense_6/bias/Read/ReadVariableOp"dense_7/kernel/Read/ReadVariableOp dense_7/bias/Read/ReadVariableOp"dense_8/kernel/Read/ReadVariableOp dense_8/bias/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOpConst*5
Tin.
,2**
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *'
f"R 
__inference__traced_save_93686


StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameconv2d_10/kernelconv2d_10/biasbatch_normalization_10/gammabatch_normalization_10/beta"batch_normalization_10/moving_mean&batch_normalization_10/moving_varianceconv2d_11/kernelconv2d_11/biasbatch_normalization_11/gammabatch_normalization_11/beta"batch_normalization_11/moving_mean&batch_normalization_11/moving_varianceconv2d_12/kernelconv2d_12/biasbatch_normalization_12/gammabatch_normalization_12/beta"batch_normalization_12/moving_mean&batch_normalization_12/moving_varianceconv2d_13/kernelconv2d_13/biasbatch_normalization_13/gammabatch_normalization_13/beta"batch_normalization_13/moving_mean&batch_normalization_13/moving_varianceconv2d_14/kernelconv2d_14/biasbatch_normalization_14/gammabatch_normalization_14/beta"batch_normalization_14/moving_mean&batch_normalization_14/moving_variancedense_6/kerneldense_6/biasdense_7/kerneldense_7/biasdense_8/kerneldense_8/biastotal_1count_1totalcount*4
Tin-
+2)*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 **
f%R#
!__inference__traced_restore_93816
ฬ

Q__inference_batch_normalization_10_layer_call_and_return_conditional_losses_91146

inputs%
readvariableop_resource:`'
readvariableop_1_resource:`6
(fusedbatchnormv3_readvariableop_resource:`8
*fusedbatchnormv3_readvariableop_1_resource:`
identityขFusedBatchNormV3/ReadVariableOpข!FusedBatchNormV3/ReadVariableOp_1ขReadVariableOpขReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:`*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:`*
dtype0
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:`*
dtype0
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:`*
dtype0ศ
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????`:`:`:`:`:*
epsilon%o:*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????`ฐ
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????`: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????`
 
_user_specified_nameinputs


D__inference_conv2d_12_layer_call_and_return_conditional_losses_91562

inputs:
conv2d_readvariableop_resource:.
biasadd_readvariableop_resource:	
identityขBiasAdd/ReadVariableOpขConv2D/ReadVariableOp~
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:*
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????*
paddingSAME*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????Y
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:?????????j
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:?????????w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :?????????	: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:?????????	
 
_user_specified_nameinputs
?
?
Q__inference_batch_normalization_14_layer_call_and_return_conditional_losses_93390

inputs&
readvariableop_resource:	(
readvariableop_1_resource:	7
(fusedbatchnormv3_readvariableop_resource:	9
*fusedbatchnormv3_readvariableop_1_resource:	
identityขFusedBatchNormV3/ReadVariableOpข!FusedBatchNormV3/ReadVariableOp_1ขReadVariableOpขReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:*
dtype0
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype0
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype0อ
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,???????????????????????????:::::*
epsilon%o:*
is_training( ~
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,???????????????????????????ฐ
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,???????????????????????????: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,???????????????????????????
 
_user_specified_nameinputs
ฤ
๊
,__inference_sequential_2_layer_call_fn_92592

inputs!
unknown:`
	unknown_0:`
	unknown_1:`
	unknown_2:`
	unknown_3:`
	unknown_4:`$
	unknown_5:`
	unknown_6:	
	unknown_7:	
	unknown_8:	
	unknown_9:	

unknown_10:	&

unknown_11:

unknown_12:	

unknown_13:	

unknown_14:	

unknown_15:	

unknown_16:	&

unknown_17:

unknown_18:	

unknown_19:	

unknown_20:	

unknown_21:	

unknown_22:	&

unknown_23:

unknown_24:	

unknown_25:	

unknown_26:	

unknown_27:	

unknown_28:	

unknown_29:


unknown_30:	

unknown_31:


unknown_32:	

unknown_33:	

unknown_34:
identityขStatefulPartitionedCallฏ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34*0
Tin)
'2%*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*F
_read_only_resource_inputs(
&$	
 !"#$*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_sequential_2_layer_call_and_return_conditional_losses_91704o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*x
_input_shapesg
e:?????????เ: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:?????????เ
 
_user_specified_nameinputs
?
?
Q__inference_batch_normalization_11_layer_call_and_return_conditional_losses_93134

inputs&
readvariableop_resource:	(
readvariableop_1_resource:	7
(fusedbatchnormv3_readvariableop_resource:	9
*fusedbatchnormv3_readvariableop_1_resource:	
identityขFusedBatchNormV3/ReadVariableOpข!FusedBatchNormV3/ReadVariableOp_1ขReadVariableOpขReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:*
dtype0
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype0
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype0อ
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,???????????????????????????:::::*
epsilon%o:*
is_training( ~
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,???????????????????????????ฐ
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,???????????????????????????: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,???????????????????????????
 
_user_specified_nameinputs
๘
b
F__inference_rescaling_2_layer_call_and_return_conditional_losses_91495

inputs
identityK
Cast/xConst*
_output_shapes
: *
dtype0*
valueB
 *;M
Cast_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *    _
mulMulinputsCast/x:output:0*
T0*1
_output_shapes
:?????????เd
addAddV2mul:z:0Cast_1/x:output:0*
T0*1
_output_shapes
:?????????เY
IdentityIdentityadd:z:0*
T0*1
_output_shapes
:?????????เ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:?????????เ:Y U
1
_output_shapes
:?????????เ
 
_user_specified_nameinputs
๑
ก
)__inference_conv2d_12_layer_call_fn_93171

inputs#
unknown:
	unknown_0:	
identityขStatefulPartitionedCallโ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_conv2d_12_layer_call_and_return_conditional_losses_91562x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :?????????	: : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:?????????	
 
_user_specified_nameinputs
?
?
Q__inference_batch_normalization_12_layer_call_and_return_conditional_losses_93226

inputs&
readvariableop_resource:	(
readvariableop_1_resource:	7
(fusedbatchnormv3_readvariableop_resource:	9
*fusedbatchnormv3_readvariableop_1_resource:	
identityขFusedBatchNormV3/ReadVariableOpข!FusedBatchNormV3/ReadVariableOp_1ขReadVariableOpขReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:*
dtype0
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype0
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype0อ
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,???????????????????????????:::::*
epsilon%o:*
is_training( ~
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,???????????????????????????ฐ
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,???????????????????????????: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,???????????????????????????
 
_user_specified_nameinputs
๓
b
)__inference_dropout_4_layer_call_fn_93459

inputs
identityขStatefulPartitionedCallภ
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dropout_4_layer_call_and_return_conditional_losses_91842p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:?????????22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:?????????
 
_user_specified_nameinputs
๚	
c
D__inference_dropout_4_layer_call_and_return_conditional_losses_91842

inputs
identityR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @e
dropout/MulMulinputsdropout/Const:output:0*
T0*(
_output_shapes
:?????????C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:?????????*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?ง
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:?????????p
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:?????????j
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*(
_output_shapes
:?????????Z
IdentityIdentitydropout/Mul_1:z:0*
T0*(
_output_shapes
:?????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:?????????:P L
(
_output_shapes
:?????????
 
_user_specified_nameinputs
ัe
๛
G__inference_sequential_2_layer_call_and_return_conditional_losses_92092

inputs)
conv2d_10_92000:`
conv2d_10_92002:`*
batch_normalization_10_92005:`*
batch_normalization_10_92007:`*
batch_normalization_10_92009:`*
batch_normalization_10_92011:`*
conv2d_11_92015:`
conv2d_11_92017:	+
batch_normalization_11_92020:	+
batch_normalization_11_92022:	+
batch_normalization_11_92024:	+
batch_normalization_11_92026:	+
conv2d_12_92030:
conv2d_12_92032:	+
batch_normalization_12_92035:	+
batch_normalization_12_92037:	+
batch_normalization_12_92039:	+
batch_normalization_12_92041:	+
conv2d_13_92044:
conv2d_13_92046:	+
batch_normalization_13_92049:	+
batch_normalization_13_92051:	+
batch_normalization_13_92053:	+
batch_normalization_13_92055:	+
conv2d_14_92058:
conv2d_14_92060:	+
batch_normalization_14_92063:	+
batch_normalization_14_92065:	+
batch_normalization_14_92067:	+
batch_normalization_14_92069:	!
dense_6_92074:

dense_6_92076:	!
dense_7_92080:

dense_7_92082:	 
dense_8_92086:	
dense_8_92088:
identityข.batch_normalization_10/StatefulPartitionedCallข.batch_normalization_11/StatefulPartitionedCallข.batch_normalization_12/StatefulPartitionedCallข.batch_normalization_13/StatefulPartitionedCallข.batch_normalization_14/StatefulPartitionedCallข!conv2d_10/StatefulPartitionedCallข!conv2d_11/StatefulPartitionedCallข!conv2d_12/StatefulPartitionedCallข!conv2d_13/StatefulPartitionedCallข!conv2d_14/StatefulPartitionedCallขdense_6/StatefulPartitionedCallขdense_7/StatefulPartitionedCallขdense_8/StatefulPartitionedCallข!dropout_4/StatefulPartitionedCallข!dropout_5/StatefulPartitionedCallว
rescaling_2/PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:?????????เ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_rescaling_2_layer_call_and_return_conditional_losses_91495
!conv2d_10/StatefulPartitionedCallStatefulPartitionedCall$rescaling_2/PartitionedCall:output:0conv2d_10_92000conv2d_10_92002*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????v`*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_conv2d_10_layer_call_and_return_conditional_losses_91508
.batch_normalization_10/StatefulPartitionedCallStatefulPartitionedCall*conv2d_10/StatefulPartitionedCall:output:0batch_normalization_10_92005batch_normalization_10_92007batch_normalization_10_92009batch_normalization_10_92011*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????v`*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Z
fURS
Q__inference_batch_normalization_10_layer_call_and_return_conditional_losses_91177?
max_pooling2d_6/PartitionedCallPartitionedCall7batch_normalization_10/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????:N`* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *S
fNRL
J__inference_max_pooling2d_6_layer_call_and_return_conditional_losses_91197
!conv2d_11/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_6/PartitionedCall:output:0conv2d_11_92015conv2d_11_92017*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_conv2d_11_layer_call_and_return_conditional_losses_91535
.batch_normalization_11/StatefulPartitionedCallStatefulPartitionedCall*conv2d_11/StatefulPartitionedCall:output:0batch_normalization_11_92020batch_normalization_11_92022batch_normalization_11_92024batch_normalization_11_92026*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Z
fURS
Q__inference_batch_normalization_11_layer_call_and_return_conditional_losses_91253?
max_pooling2d_7/PartitionedCallPartitionedCall7batch_normalization_11/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????	* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *S
fNRL
J__inference_max_pooling2d_7_layer_call_and_return_conditional_losses_91273
!conv2d_12/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_7/PartitionedCall:output:0conv2d_12_92030conv2d_12_92032*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_conv2d_12_layer_call_and_return_conditional_losses_91562
.batch_normalization_12/StatefulPartitionedCallStatefulPartitionedCall*conv2d_12/StatefulPartitionedCall:output:0batch_normalization_12_92035batch_normalization_12_92037batch_normalization_12_92039batch_normalization_12_92041*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Z
fURS
Q__inference_batch_normalization_12_layer_call_and_return_conditional_losses_91329ซ
!conv2d_13/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_12/StatefulPartitionedCall:output:0conv2d_13_92044conv2d_13_92046*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_conv2d_13_layer_call_and_return_conditional_losses_91588
.batch_normalization_13/StatefulPartitionedCallStatefulPartitionedCall*conv2d_13/StatefulPartitionedCall:output:0batch_normalization_13_92049batch_normalization_13_92051batch_normalization_13_92053batch_normalization_13_92055*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Z
fURS
Q__inference_batch_normalization_13_layer_call_and_return_conditional_losses_91393ซ
!conv2d_14/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_13/StatefulPartitionedCall:output:0conv2d_14_92058conv2d_14_92060*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_conv2d_14_layer_call_and_return_conditional_losses_91614
.batch_normalization_14/StatefulPartitionedCallStatefulPartitionedCall*conv2d_14/StatefulPartitionedCall:output:0batch_normalization_14_92063batch_normalization_14_92065batch_normalization_14_92067batch_normalization_14_92069*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Z
fURS
Q__inference_batch_normalization_14_layer_call_and_return_conditional_losses_91457?
max_pooling2d_8/PartitionedCallPartitionedCall7batch_normalization_14/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *S
fNRL
J__inference_max_pooling2d_8_layer_call_and_return_conditional_losses_91477?
flatten_2/PartitionedCallPartitionedCall(max_pooling2d_8/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_flatten_2_layer_call_and_return_conditional_losses_91636
dense_6/StatefulPartitionedCallStatefulPartitionedCall"flatten_2/PartitionedCall:output:0dense_6_92074dense_6_92076*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_dense_6_layer_call_and_return_conditional_losses_91649์
!dropout_4/StatefulPartitionedCallStatefulPartitionedCall(dense_6/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dropout_4_layer_call_and_return_conditional_losses_91842
dense_7/StatefulPartitionedCallStatefulPartitionedCall*dropout_4/StatefulPartitionedCall:output:0dense_7_92080dense_7_92082*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_dense_7_layer_call_and_return_conditional_losses_91673
!dropout_5/StatefulPartitionedCallStatefulPartitionedCall(dense_7/StatefulPartitionedCall:output:0"^dropout_4/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dropout_5_layer_call_and_return_conditional_losses_91809
dense_8/StatefulPartitionedCallStatefulPartitionedCall*dropout_5/StatefulPartitionedCall:output:0dense_8_92086dense_8_92088*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_dense_8_layer_call_and_return_conditional_losses_91697w
IdentityIdentity(dense_8/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????
NoOpNoOp/^batch_normalization_10/StatefulPartitionedCall/^batch_normalization_11/StatefulPartitionedCall/^batch_normalization_12/StatefulPartitionedCall/^batch_normalization_13/StatefulPartitionedCall/^batch_normalization_14/StatefulPartitionedCall"^conv2d_10/StatefulPartitionedCall"^conv2d_11/StatefulPartitionedCall"^conv2d_12/StatefulPartitionedCall"^conv2d_13/StatefulPartitionedCall"^conv2d_14/StatefulPartitionedCall ^dense_6/StatefulPartitionedCall ^dense_7/StatefulPartitionedCall ^dense_8/StatefulPartitionedCall"^dropout_4/StatefulPartitionedCall"^dropout_5/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*x
_input_shapesg
e:?????????เ: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2`
.batch_normalization_10/StatefulPartitionedCall.batch_normalization_10/StatefulPartitionedCall2`
.batch_normalization_11/StatefulPartitionedCall.batch_normalization_11/StatefulPartitionedCall2`
.batch_normalization_12/StatefulPartitionedCall.batch_normalization_12/StatefulPartitionedCall2`
.batch_normalization_13/StatefulPartitionedCall.batch_normalization_13/StatefulPartitionedCall2`
.batch_normalization_14/StatefulPartitionedCall.batch_normalization_14/StatefulPartitionedCall2F
!conv2d_10/StatefulPartitionedCall!conv2d_10/StatefulPartitionedCall2F
!conv2d_11/StatefulPartitionedCall!conv2d_11/StatefulPartitionedCall2F
!conv2d_12/StatefulPartitionedCall!conv2d_12/StatefulPartitionedCall2F
!conv2d_13/StatefulPartitionedCall!conv2d_13/StatefulPartitionedCall2F
!conv2d_14/StatefulPartitionedCall!conv2d_14/StatefulPartitionedCall2B
dense_6/StatefulPartitionedCalldense_6/StatefulPartitionedCall2B
dense_7/StatefulPartitionedCalldense_7/StatefulPartitionedCall2B
dense_8/StatefulPartitionedCalldense_8/StatefulPartitionedCall2F
!dropout_4/StatefulPartitionedCall!dropout_4/StatefulPartitionedCall2F
!dropout_5/StatefulPartitionedCall!dropout_5/StatefulPartitionedCall:Y U
1
_output_shapes
:?????????เ
 
_user_specified_nameinputs
๑
ก
)__inference_conv2d_14_layer_call_fn_93335

inputs#
unknown:
	unknown_0:	
identityขStatefulPartitionedCallโ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_conv2d_14_layer_call_and_return_conditional_losses_91614x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :?????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:?????????
 
_user_specified_nameinputs
c
พ
G__inference_sequential_2_layer_call_and_return_conditional_losses_92340
rescaling_2_input)
conv2d_10_92248:`
conv2d_10_92250:`*
batch_normalization_10_92253:`*
batch_normalization_10_92255:`*
batch_normalization_10_92257:`*
batch_normalization_10_92259:`*
conv2d_11_92263:`
conv2d_11_92265:	+
batch_normalization_11_92268:	+
batch_normalization_11_92270:	+
batch_normalization_11_92272:	+
batch_normalization_11_92274:	+
conv2d_12_92278:
conv2d_12_92280:	+
batch_normalization_12_92283:	+
batch_normalization_12_92285:	+
batch_normalization_12_92287:	+
batch_normalization_12_92289:	+
conv2d_13_92292:
conv2d_13_92294:	+
batch_normalization_13_92297:	+
batch_normalization_13_92299:	+
batch_normalization_13_92301:	+
batch_normalization_13_92303:	+
conv2d_14_92306:
conv2d_14_92308:	+
batch_normalization_14_92311:	+
batch_normalization_14_92313:	+
batch_normalization_14_92315:	+
batch_normalization_14_92317:	!
dense_6_92322:

dense_6_92324:	!
dense_7_92328:

dense_7_92330:	 
dense_8_92334:	
dense_8_92336:
identityข.batch_normalization_10/StatefulPartitionedCallข.batch_normalization_11/StatefulPartitionedCallข.batch_normalization_12/StatefulPartitionedCallข.batch_normalization_13/StatefulPartitionedCallข.batch_normalization_14/StatefulPartitionedCallข!conv2d_10/StatefulPartitionedCallข!conv2d_11/StatefulPartitionedCallข!conv2d_12/StatefulPartitionedCallข!conv2d_13/StatefulPartitionedCallข!conv2d_14/StatefulPartitionedCallขdense_6/StatefulPartitionedCallขdense_7/StatefulPartitionedCallขdense_8/StatefulPartitionedCallา
rescaling_2/PartitionedCallPartitionedCallrescaling_2_input*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:?????????เ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_rescaling_2_layer_call_and_return_conditional_losses_91495
!conv2d_10/StatefulPartitionedCallStatefulPartitionedCall$rescaling_2/PartitionedCall:output:0conv2d_10_92248conv2d_10_92250*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????v`*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_conv2d_10_layer_call_and_return_conditional_losses_91508
.batch_normalization_10/StatefulPartitionedCallStatefulPartitionedCall*conv2d_10/StatefulPartitionedCall:output:0batch_normalization_10_92253batch_normalization_10_92255batch_normalization_10_92257batch_normalization_10_92259*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????v`*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Z
fURS
Q__inference_batch_normalization_10_layer_call_and_return_conditional_losses_91146?
max_pooling2d_6/PartitionedCallPartitionedCall7batch_normalization_10/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????:N`* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *S
fNRL
J__inference_max_pooling2d_6_layer_call_and_return_conditional_losses_91197
!conv2d_11/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_6/PartitionedCall:output:0conv2d_11_92263conv2d_11_92265*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_conv2d_11_layer_call_and_return_conditional_losses_91535
.batch_normalization_11/StatefulPartitionedCallStatefulPartitionedCall*conv2d_11/StatefulPartitionedCall:output:0batch_normalization_11_92268batch_normalization_11_92270batch_normalization_11_92272batch_normalization_11_92274*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Z
fURS
Q__inference_batch_normalization_11_layer_call_and_return_conditional_losses_91222?
max_pooling2d_7/PartitionedCallPartitionedCall7batch_normalization_11/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????	* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *S
fNRL
J__inference_max_pooling2d_7_layer_call_and_return_conditional_losses_91273
!conv2d_12/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_7/PartitionedCall:output:0conv2d_12_92278conv2d_12_92280*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_conv2d_12_layer_call_and_return_conditional_losses_91562
.batch_normalization_12/StatefulPartitionedCallStatefulPartitionedCall*conv2d_12/StatefulPartitionedCall:output:0batch_normalization_12_92283batch_normalization_12_92285batch_normalization_12_92287batch_normalization_12_92289*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Z
fURS
Q__inference_batch_normalization_12_layer_call_and_return_conditional_losses_91298ซ
!conv2d_13/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_12/StatefulPartitionedCall:output:0conv2d_13_92292conv2d_13_92294*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_conv2d_13_layer_call_and_return_conditional_losses_91588
.batch_normalization_13/StatefulPartitionedCallStatefulPartitionedCall*conv2d_13/StatefulPartitionedCall:output:0batch_normalization_13_92297batch_normalization_13_92299batch_normalization_13_92301batch_normalization_13_92303*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Z
fURS
Q__inference_batch_normalization_13_layer_call_and_return_conditional_losses_91362ซ
!conv2d_14/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_13/StatefulPartitionedCall:output:0conv2d_14_92306conv2d_14_92308*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_conv2d_14_layer_call_and_return_conditional_losses_91614
.batch_normalization_14/StatefulPartitionedCallStatefulPartitionedCall*conv2d_14/StatefulPartitionedCall:output:0batch_normalization_14_92311batch_normalization_14_92313batch_normalization_14_92315batch_normalization_14_92317*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Z
fURS
Q__inference_batch_normalization_14_layer_call_and_return_conditional_losses_91426?
max_pooling2d_8/PartitionedCallPartitionedCall7batch_normalization_14/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *S
fNRL
J__inference_max_pooling2d_8_layer_call_and_return_conditional_losses_91477?
flatten_2/PartitionedCallPartitionedCall(max_pooling2d_8/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_flatten_2_layer_call_and_return_conditional_losses_91636
dense_6/StatefulPartitionedCallStatefulPartitionedCall"flatten_2/PartitionedCall:output:0dense_6_92322dense_6_92324*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_dense_6_layer_call_and_return_conditional_losses_91649?
dropout_4/PartitionedCallPartitionedCall(dense_6/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dropout_4_layer_call_and_return_conditional_losses_91660
dense_7/StatefulPartitionedCallStatefulPartitionedCall"dropout_4/PartitionedCall:output:0dense_7_92328dense_7_92330*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_dense_7_layer_call_and_return_conditional_losses_91673?
dropout_5/PartitionedCallPartitionedCall(dense_7/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dropout_5_layer_call_and_return_conditional_losses_91684
dense_8/StatefulPartitionedCallStatefulPartitionedCall"dropout_5/PartitionedCall:output:0dense_8_92334dense_8_92336*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_dense_8_layer_call_and_return_conditional_losses_91697w
IdentityIdentity(dense_8/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????ี
NoOpNoOp/^batch_normalization_10/StatefulPartitionedCall/^batch_normalization_11/StatefulPartitionedCall/^batch_normalization_12/StatefulPartitionedCall/^batch_normalization_13/StatefulPartitionedCall/^batch_normalization_14/StatefulPartitionedCall"^conv2d_10/StatefulPartitionedCall"^conv2d_11/StatefulPartitionedCall"^conv2d_12/StatefulPartitionedCall"^conv2d_13/StatefulPartitionedCall"^conv2d_14/StatefulPartitionedCall ^dense_6/StatefulPartitionedCall ^dense_7/StatefulPartitionedCall ^dense_8/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*x
_input_shapesg
e:?????????เ: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2`
.batch_normalization_10/StatefulPartitionedCall.batch_normalization_10/StatefulPartitionedCall2`
.batch_normalization_11/StatefulPartitionedCall.batch_normalization_11/StatefulPartitionedCall2`
.batch_normalization_12/StatefulPartitionedCall.batch_normalization_12/StatefulPartitionedCall2`
.batch_normalization_13/StatefulPartitionedCall.batch_normalization_13/StatefulPartitionedCall2`
.batch_normalization_14/StatefulPartitionedCall.batch_normalization_14/StatefulPartitionedCall2F
!conv2d_10/StatefulPartitionedCall!conv2d_10/StatefulPartitionedCall2F
!conv2d_11/StatefulPartitionedCall!conv2d_11/StatefulPartitionedCall2F
!conv2d_12/StatefulPartitionedCall!conv2d_12/StatefulPartitionedCall2F
!conv2d_13/StatefulPartitionedCall!conv2d_13/StatefulPartitionedCall2F
!conv2d_14/StatefulPartitionedCall!conv2d_14/StatefulPartitionedCall2B
dense_6/StatefulPartitionedCalldense_6/StatefulPartitionedCall2B
dense_7/StatefulPartitionedCalldense_7/StatefulPartitionedCall2B
dense_8/StatefulPartitionedCalldense_8/StatefulPartitionedCall:d `
1
_output_shapes
:?????????เ
+
_user_specified_namerescaling_2_input
ศ
`
D__inference_flatten_2_layer_call_and_return_conditional_losses_91636

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"????   ]
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:?????????Y
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:?????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:?????????:X T
0
_output_shapes
:?????????
 
_user_specified_nameinputs

f
J__inference_max_pooling2d_6_layer_call_and_return_conditional_losses_91197

inputs
identityข
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4????????????????????????????????????*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
	
ั
6__inference_batch_normalization_10_layer_call_fn_93011

inputs
unknown:`
	unknown_0:`
	unknown_1:`
	unknown_2:`
identityขStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????`*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Z
fURS
Q__inference_batch_normalization_10_layer_call_and_return_conditional_losses_91146
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????``
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????`: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+???????????????????????????`
 
_user_specified_nameinputs
ล

'__inference_dense_6_layer_call_fn_93438

inputs
unknown:

	unknown_0:	
identityขStatefulPartitionedCallุ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_dense_6_layer_call_and_return_conditional_losses_91649p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:?????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:?????????
 
_user_specified_nameinputs
็b
ณ
G__inference_sequential_2_layer_call_and_return_conditional_losses_91704

inputs)
conv2d_10_91509:`
conv2d_10_91511:`*
batch_normalization_10_91514:`*
batch_normalization_10_91516:`*
batch_normalization_10_91518:`*
batch_normalization_10_91520:`*
conv2d_11_91536:`
conv2d_11_91538:	+
batch_normalization_11_91541:	+
batch_normalization_11_91543:	+
batch_normalization_11_91545:	+
batch_normalization_11_91547:	+
conv2d_12_91563:
conv2d_12_91565:	+
batch_normalization_12_91568:	+
batch_normalization_12_91570:	+
batch_normalization_12_91572:	+
batch_normalization_12_91574:	+
conv2d_13_91589:
conv2d_13_91591:	+
batch_normalization_13_91594:	+
batch_normalization_13_91596:	+
batch_normalization_13_91598:	+
batch_normalization_13_91600:	+
conv2d_14_91615:
conv2d_14_91617:	+
batch_normalization_14_91620:	+
batch_normalization_14_91622:	+
batch_normalization_14_91624:	+
batch_normalization_14_91626:	!
dense_6_91650:

dense_6_91652:	!
dense_7_91674:

dense_7_91676:	 
dense_8_91698:	
dense_8_91700:
identityข.batch_normalization_10/StatefulPartitionedCallข.batch_normalization_11/StatefulPartitionedCallข.batch_normalization_12/StatefulPartitionedCallข.batch_normalization_13/StatefulPartitionedCallข.batch_normalization_14/StatefulPartitionedCallข!conv2d_10/StatefulPartitionedCallข!conv2d_11/StatefulPartitionedCallข!conv2d_12/StatefulPartitionedCallข!conv2d_13/StatefulPartitionedCallข!conv2d_14/StatefulPartitionedCallขdense_6/StatefulPartitionedCallขdense_7/StatefulPartitionedCallขdense_8/StatefulPartitionedCallว
rescaling_2/PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:?????????เ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_rescaling_2_layer_call_and_return_conditional_losses_91495
!conv2d_10/StatefulPartitionedCallStatefulPartitionedCall$rescaling_2/PartitionedCall:output:0conv2d_10_91509conv2d_10_91511*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????v`*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_conv2d_10_layer_call_and_return_conditional_losses_91508
.batch_normalization_10/StatefulPartitionedCallStatefulPartitionedCall*conv2d_10/StatefulPartitionedCall:output:0batch_normalization_10_91514batch_normalization_10_91516batch_normalization_10_91518batch_normalization_10_91520*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????v`*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Z
fURS
Q__inference_batch_normalization_10_layer_call_and_return_conditional_losses_91146?
max_pooling2d_6/PartitionedCallPartitionedCall7batch_normalization_10/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????:N`* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *S
fNRL
J__inference_max_pooling2d_6_layer_call_and_return_conditional_losses_91197
!conv2d_11/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_6/PartitionedCall:output:0conv2d_11_91536conv2d_11_91538*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_conv2d_11_layer_call_and_return_conditional_losses_91535
.batch_normalization_11/StatefulPartitionedCallStatefulPartitionedCall*conv2d_11/StatefulPartitionedCall:output:0batch_normalization_11_91541batch_normalization_11_91543batch_normalization_11_91545batch_normalization_11_91547*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Z
fURS
Q__inference_batch_normalization_11_layer_call_and_return_conditional_losses_91222?
max_pooling2d_7/PartitionedCallPartitionedCall7batch_normalization_11/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????	* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *S
fNRL
J__inference_max_pooling2d_7_layer_call_and_return_conditional_losses_91273
!conv2d_12/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_7/PartitionedCall:output:0conv2d_12_91563conv2d_12_91565*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_conv2d_12_layer_call_and_return_conditional_losses_91562
.batch_normalization_12/StatefulPartitionedCallStatefulPartitionedCall*conv2d_12/StatefulPartitionedCall:output:0batch_normalization_12_91568batch_normalization_12_91570batch_normalization_12_91572batch_normalization_12_91574*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Z
fURS
Q__inference_batch_normalization_12_layer_call_and_return_conditional_losses_91298ซ
!conv2d_13/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_12/StatefulPartitionedCall:output:0conv2d_13_91589conv2d_13_91591*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_conv2d_13_layer_call_and_return_conditional_losses_91588
.batch_normalization_13/StatefulPartitionedCallStatefulPartitionedCall*conv2d_13/StatefulPartitionedCall:output:0batch_normalization_13_91594batch_normalization_13_91596batch_normalization_13_91598batch_normalization_13_91600*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Z
fURS
Q__inference_batch_normalization_13_layer_call_and_return_conditional_losses_91362ซ
!conv2d_14/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_13/StatefulPartitionedCall:output:0conv2d_14_91615conv2d_14_91617*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_conv2d_14_layer_call_and_return_conditional_losses_91614
.batch_normalization_14/StatefulPartitionedCallStatefulPartitionedCall*conv2d_14/StatefulPartitionedCall:output:0batch_normalization_14_91620batch_normalization_14_91622batch_normalization_14_91624batch_normalization_14_91626*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Z
fURS
Q__inference_batch_normalization_14_layer_call_and_return_conditional_losses_91426?
max_pooling2d_8/PartitionedCallPartitionedCall7batch_normalization_14/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *S
fNRL
J__inference_max_pooling2d_8_layer_call_and_return_conditional_losses_91477?
flatten_2/PartitionedCallPartitionedCall(max_pooling2d_8/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_flatten_2_layer_call_and_return_conditional_losses_91636
dense_6/StatefulPartitionedCallStatefulPartitionedCall"flatten_2/PartitionedCall:output:0dense_6_91650dense_6_91652*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_dense_6_layer_call_and_return_conditional_losses_91649?
dropout_4/PartitionedCallPartitionedCall(dense_6/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dropout_4_layer_call_and_return_conditional_losses_91660
dense_7/StatefulPartitionedCallStatefulPartitionedCall"dropout_4/PartitionedCall:output:0dense_7_91674dense_7_91676*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_dense_7_layer_call_and_return_conditional_losses_91673?
dropout_5/PartitionedCallPartitionedCall(dense_7/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dropout_5_layer_call_and_return_conditional_losses_91684
dense_8/StatefulPartitionedCallStatefulPartitionedCall"dropout_5/PartitionedCall:output:0dense_8_91698dense_8_91700*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_dense_8_layer_call_and_return_conditional_losses_91697w
IdentityIdentity(dense_8/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????ี
NoOpNoOp/^batch_normalization_10/StatefulPartitionedCall/^batch_normalization_11/StatefulPartitionedCall/^batch_normalization_12/StatefulPartitionedCall/^batch_normalization_13/StatefulPartitionedCall/^batch_normalization_14/StatefulPartitionedCall"^conv2d_10/StatefulPartitionedCall"^conv2d_11/StatefulPartitionedCall"^conv2d_12/StatefulPartitionedCall"^conv2d_13/StatefulPartitionedCall"^conv2d_14/StatefulPartitionedCall ^dense_6/StatefulPartitionedCall ^dense_7/StatefulPartitionedCall ^dense_8/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*x
_input_shapesg
e:?????????เ: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2`
.batch_normalization_10/StatefulPartitionedCall.batch_normalization_10/StatefulPartitionedCall2`
.batch_normalization_11/StatefulPartitionedCall.batch_normalization_11/StatefulPartitionedCall2`
.batch_normalization_12/StatefulPartitionedCall.batch_normalization_12/StatefulPartitionedCall2`
.batch_normalization_13/StatefulPartitionedCall.batch_normalization_13/StatefulPartitionedCall2`
.batch_normalization_14/StatefulPartitionedCall.batch_normalization_14/StatefulPartitionedCall2F
!conv2d_10/StatefulPartitionedCall!conv2d_10/StatefulPartitionedCall2F
!conv2d_11/StatefulPartitionedCall!conv2d_11/StatefulPartitionedCall2F
!conv2d_12/StatefulPartitionedCall!conv2d_12/StatefulPartitionedCall2F
!conv2d_13/StatefulPartitionedCall!conv2d_13/StatefulPartitionedCall2F
!conv2d_14/StatefulPartitionedCall!conv2d_14/StatefulPartitionedCall2B
dense_6/StatefulPartitionedCalldense_6/StatefulPartitionedCall2B
dense_7/StatefulPartitionedCalldense_7/StatefulPartitionedCall2B
dense_8/StatefulPartitionedCalldense_8/StatefulPartitionedCall:Y U
1
_output_shapes
:?????????เ
 
_user_specified_nameinputs
	
ี
6__inference_batch_normalization_14_layer_call_fn_93359

inputs
unknown:	
	unknown_0:	
	unknown_1:	
	unknown_2:	
identityขStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,???????????????????????????*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Z
fURS
Q__inference_batch_normalization_14_layer_call_and_return_conditional_losses_91426
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*B
_output_shapes0
.:,???????????????????????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,???????????????????????????: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,???????????????????????????
 
_user_specified_nameinputs
?
?
Q__inference_batch_normalization_11_layer_call_and_return_conditional_losses_91222

inputs&
readvariableop_resource:	(
readvariableop_1_resource:	7
(fusedbatchnormv3_readvariableop_resource:	9
*fusedbatchnormv3_readvariableop_1_resource:	
identityขFusedBatchNormV3/ReadVariableOpข!FusedBatchNormV3/ReadVariableOp_1ขReadVariableOpขReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:*
dtype0
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype0
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype0อ
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,???????????????????????????:::::*
epsilon%o:*
is_training( ~
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,???????????????????????????ฐ
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,???????????????????????????: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,???????????????????????????
 
_user_specified_nameinputs

f
J__inference_max_pooling2d_7_layer_call_and_return_conditional_losses_91273

inputs
identityข
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4????????????????????????????????????*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
ฬ

Q__inference_batch_normalization_10_layer_call_and_return_conditional_losses_93042

inputs%
readvariableop_resource:`'
readvariableop_1_resource:`6
(fusedbatchnormv3_readvariableop_resource:`8
*fusedbatchnormv3_readvariableop_1_resource:`
identityขFusedBatchNormV3/ReadVariableOpข!FusedBatchNormV3/ReadVariableOp_1ขReadVariableOpขReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:`*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:`*
dtype0
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:`*
dtype0
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:`*
dtype0ศ
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????`:`:`:`:`:*
epsilon%o:*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????`ฐ
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????`: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????`
 
_user_specified_nameinputs


D__inference_conv2d_13_layer_call_and_return_conditional_losses_93264

inputs:
conv2d_readvariableop_resource:.
biasadd_readvariableop_resource:	
identityขBiasAdd/ReadVariableOpขConv2D/ReadVariableOp~
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:*
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????*
paddingSAME*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????Y
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:?????????j
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:?????????w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :?????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:?????????
 
_user_specified_nameinputs

ฤ
Q__inference_batch_normalization_13_layer_call_and_return_conditional_losses_91393

inputs&
readvariableop_resource:	(
readvariableop_1_resource:	7
(fusedbatchnormv3_readvariableop_resource:	9
*fusedbatchnormv3_readvariableop_1_resource:	
identityขAssignNewValueขAssignNewValue_1ขFusedBatchNormV3/ReadVariableOpข!FusedBatchNormV3/ReadVariableOp_1ขReadVariableOpขReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:*
dtype0
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype0
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype0?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,???????????????????????????:::::*
epsilon%o:*
exponential_avg_factor%
ื#<ฦ
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(ะ
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(~
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,???????????????????????????ิ
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,???????????????????????????: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,???????????????????????????
 
_user_specified_nameinputs
๎
?
)__inference_conv2d_11_layer_call_fn_93079

inputs"
unknown:`
	unknown_0:	
identityขStatefulPartitionedCallโ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_conv2d_11_layer_call_and_return_conditional_losses_91535x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????:N`: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????:N`
 
_user_specified_nameinputs
	
ี
6__inference_batch_normalization_13_layer_call_fn_93277

inputs
unknown:	
	unknown_0:	
	unknown_1:	
	unknown_2:	
identityขStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,???????????????????????????*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Z
fURS
Q__inference_batch_normalization_13_layer_call_and_return_conditional_losses_91362
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*B
_output_shapes0
.:,???????????????????????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,???????????????????????????: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,???????????????????????????
 
_user_specified_nameinputs
๚	
c
D__inference_dropout_5_layer_call_and_return_conditional_losses_93523

inputs
identityR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @e
dropout/MulMulinputsdropout/Const:output:0*
T0*(
_output_shapes
:?????????C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:?????????*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?ง
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:?????????p
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:?????????j
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*(
_output_shapes
:?????????Z
IdentityIdentitydropout/Mul_1:z:0*
T0*(
_output_shapes
:?????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:?????????:P L
(
_output_shapes
:?????????
 
_user_specified_nameinputs
ศ
`
D__inference_flatten_2_layer_call_and_return_conditional_losses_93429

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"????   ]
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:?????????Y
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:?????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:?????????:X T
0
_output_shapes
:?????????
 
_user_specified_nameinputs


D__inference_conv2d_12_layer_call_and_return_conditional_losses_93182

inputs:
conv2d_readvariableop_resource:.
biasadd_readvariableop_resource:	
identityขBiasAdd/ReadVariableOpขConv2D/ReadVariableOp~
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:*
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????*
paddingSAME*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????Y
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:?????????j
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:?????????w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :?????????	: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:?????????	
 
_user_specified_nameinputs
?
?
Q__inference_batch_normalization_13_layer_call_and_return_conditional_losses_93308

inputs&
readvariableop_resource:	(
readvariableop_1_resource:	7
(fusedbatchnormv3_readvariableop_resource:	9
*fusedbatchnormv3_readvariableop_1_resource:	
identityขFusedBatchNormV3/ReadVariableOpข!FusedBatchNormV3/ReadVariableOp_1ขReadVariableOpขReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:*
dtype0
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype0
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype0อ
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,???????????????????????????:::::*
epsilon%o:*
is_training( ~
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,???????????????????????????ฐ
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,???????????????????????????: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,???????????????????????????
 
_user_specified_nameinputs
ฅ

๖
B__inference_dense_7_layer_call_and_return_conditional_losses_93496

inputs2
matmul_readvariableop_resource:
.
biasadd_readvariableop_resource:	
identityขBiasAdd/ReadVariableOpขMatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????Q
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:?????????b
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:?????????w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:?????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:?????????
 
_user_specified_nameinputs
	
ี
6__inference_batch_normalization_14_layer_call_fn_93372

inputs
unknown:	
	unknown_0:	
	unknown_1:	
	unknown_2:	
identityขStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,???????????????????????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Z
fURS
Q__inference_batch_normalization_14_layer_call_and_return_conditional_losses_91457
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*B
_output_shapes0
.:,???????????????????????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,???????????????????????????: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,???????????????????????????
 
_user_specified_nameinputs
ม

'__inference_dense_8_layer_call_fn_93532

inputs
unknown:	
	unknown_0:
identityขStatefulPartitionedCallื
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_dense_8_layer_call_and_return_conditional_losses_91697o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:?????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:?????????
 
_user_specified_nameinputs
๚

!__inference__traced_restore_93816
file_prefix;
!assignvariableop_conv2d_10_kernel:`/
!assignvariableop_1_conv2d_10_bias:`=
/assignvariableop_2_batch_normalization_10_gamma:`<
.assignvariableop_3_batch_normalization_10_beta:`C
5assignvariableop_4_batch_normalization_10_moving_mean:`G
9assignvariableop_5_batch_normalization_10_moving_variance:`>
#assignvariableop_6_conv2d_11_kernel:`0
!assignvariableop_7_conv2d_11_bias:	>
/assignvariableop_8_batch_normalization_11_gamma:	=
.assignvariableop_9_batch_normalization_11_beta:	E
6assignvariableop_10_batch_normalization_11_moving_mean:	I
:assignvariableop_11_batch_normalization_11_moving_variance:	@
$assignvariableop_12_conv2d_12_kernel:1
"assignvariableop_13_conv2d_12_bias:	?
0assignvariableop_14_batch_normalization_12_gamma:	>
/assignvariableop_15_batch_normalization_12_beta:	E
6assignvariableop_16_batch_normalization_12_moving_mean:	I
:assignvariableop_17_batch_normalization_12_moving_variance:	@
$assignvariableop_18_conv2d_13_kernel:1
"assignvariableop_19_conv2d_13_bias:	?
0assignvariableop_20_batch_normalization_13_gamma:	>
/assignvariableop_21_batch_normalization_13_beta:	E
6assignvariableop_22_batch_normalization_13_moving_mean:	I
:assignvariableop_23_batch_normalization_13_moving_variance:	@
$assignvariableop_24_conv2d_14_kernel:1
"assignvariableop_25_conv2d_14_bias:	?
0assignvariableop_26_batch_normalization_14_gamma:	>
/assignvariableop_27_batch_normalization_14_beta:	E
6assignvariableop_28_batch_normalization_14_moving_mean:	I
:assignvariableop_29_batch_normalization_14_moving_variance:	6
"assignvariableop_30_dense_6_kernel:
/
 assignvariableop_31_dense_6_bias:	6
"assignvariableop_32_dense_7_kernel:
/
 assignvariableop_33_dense_7_bias:	5
"assignvariableop_34_dense_8_kernel:	.
 assignvariableop_35_dense_8_bias:%
assignvariableop_36_total_1: %
assignvariableop_37_count_1: #
assignvariableop_38_total: #
assignvariableop_39_count: 
identity_41ขAssignVariableOpขAssignVariableOp_1ขAssignVariableOp_10ขAssignVariableOp_11ขAssignVariableOp_12ขAssignVariableOp_13ขAssignVariableOp_14ขAssignVariableOp_15ขAssignVariableOp_16ขAssignVariableOp_17ขAssignVariableOp_18ขAssignVariableOp_19ขAssignVariableOp_2ขAssignVariableOp_20ขAssignVariableOp_21ขAssignVariableOp_22ขAssignVariableOp_23ขAssignVariableOp_24ขAssignVariableOp_25ขAssignVariableOp_26ขAssignVariableOp_27ขAssignVariableOp_28ขAssignVariableOp_29ขAssignVariableOp_3ขAssignVariableOp_30ขAssignVariableOp_31ขAssignVariableOp_32ขAssignVariableOp_33ขAssignVariableOp_34ขAssignVariableOp_35ขAssignVariableOp_36ขAssignVariableOp_37ขAssignVariableOp_38ขAssignVariableOp_39ขAssignVariableOp_4ขAssignVariableOp_5ขAssignVariableOp_6ขAssignVariableOp_7ขAssignVariableOp_8ขAssignVariableOp_9๒
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:)*
dtype0*
valueB)B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-1/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-1/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-3/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-3/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-5/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-5/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-5/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-7/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-7/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-7/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-9/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-9/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-9/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-9/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-10/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-10/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-11/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-11/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-12/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-12/bias/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPHย
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:)*
dtype0*e
value\BZ)B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B ๎
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*บ
_output_shapesง
ค:::::::::::::::::::::::::::::::::::::::::*7
dtypes-
+2)[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOpAssignVariableOp!assignvariableop_conv2d_10_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_1AssignVariableOp!assignvariableop_1_conv2d_10_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_2AssignVariableOp/assignvariableop_2_batch_normalization_10_gammaIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_3AssignVariableOp.assignvariableop_3_batch_normalization_10_betaIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:ค
AssignVariableOp_4AssignVariableOp5assignvariableop_4_batch_normalization_10_moving_meanIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:จ
AssignVariableOp_5AssignVariableOp9assignvariableop_5_batch_normalization_10_moving_varianceIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_6AssignVariableOp#assignvariableop_6_conv2d_11_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_7AssignVariableOp!assignvariableop_7_conv2d_11_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_8AssignVariableOp/assignvariableop_8_batch_normalization_11_gammaIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_9AssignVariableOp.assignvariableop_9_batch_normalization_11_betaIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:ง
AssignVariableOp_10AssignVariableOp6assignvariableop_10_batch_normalization_11_moving_meanIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:ซ
AssignVariableOp_11AssignVariableOp:assignvariableop_11_batch_normalization_11_moving_varianceIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_12AssignVariableOp$assignvariableop_12_conv2d_12_kernelIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_13AssignVariableOp"assignvariableop_13_conv2d_12_biasIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:ก
AssignVariableOp_14AssignVariableOp0assignvariableop_14_batch_normalization_12_gammaIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_15AssignVariableOp/assignvariableop_15_batch_normalization_12_betaIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:ง
AssignVariableOp_16AssignVariableOp6assignvariableop_16_batch_normalization_12_moving_meanIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:ซ
AssignVariableOp_17AssignVariableOp:assignvariableop_17_batch_normalization_12_moving_varianceIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_18AssignVariableOp$assignvariableop_18_conv2d_13_kernelIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_19AssignVariableOp"assignvariableop_19_conv2d_13_biasIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:ก
AssignVariableOp_20AssignVariableOp0assignvariableop_20_batch_normalization_13_gammaIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_21AssignVariableOp/assignvariableop_21_batch_normalization_13_betaIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:ง
AssignVariableOp_22AssignVariableOp6assignvariableop_22_batch_normalization_13_moving_meanIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:ซ
AssignVariableOp_23AssignVariableOp:assignvariableop_23_batch_normalization_13_moving_varianceIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_24AssignVariableOp$assignvariableop_24_conv2d_14_kernelIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_25AssignVariableOp"assignvariableop_25_conv2d_14_biasIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:ก
AssignVariableOp_26AssignVariableOp0assignvariableop_26_batch_normalization_14_gammaIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_27AssignVariableOp/assignvariableop_27_batch_normalization_14_betaIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:ง
AssignVariableOp_28AssignVariableOp6assignvariableop_28_batch_normalization_14_moving_meanIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:ซ
AssignVariableOp_29AssignVariableOp:assignvariableop_29_batch_normalization_14_moving_varianceIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_30AssignVariableOp"assignvariableop_30_dense_6_kernelIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_31AssignVariableOp assignvariableop_31_dense_6_biasIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_32AssignVariableOp"assignvariableop_32_dense_7_kernelIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_33AssignVariableOp assignvariableop_33_dense_7_biasIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_34AssignVariableOp"assignvariableop_34_dense_8_kernelIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_35AssignVariableOp assignvariableop_35_dense_8_biasIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_36AssignVariableOpassignvariableop_36_total_1Identity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_37AssignVariableOpassignvariableop_37_count_1Identity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_38AssignVariableOpassignvariableop_38_totalIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_39AssignVariableOpassignvariableop_39_countIdentity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype01
NoOpNoOp"/device:CPU:0*
_output_shapes
 ฟ
Identity_40Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_41IdentityIdentity_40:output:0^NoOp_1*
T0*
_output_shapes
: ฌ
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 "#
identity_41Identity_41:output:0*e
_input_shapesT
R: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_34AssignVariableOp_342*
AssignVariableOp_35AssignVariableOp_352*
AssignVariableOp_36AssignVariableOp_362*
AssignVariableOp_37AssignVariableOp_372*
AssignVariableOp_38AssignVariableOp_382*
AssignVariableOp_39AssignVariableOp_392(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
ฅ

๖
B__inference_dense_7_layer_call_and_return_conditional_losses_91673

inputs2
matmul_readvariableop_resource:
.
biasadd_readvariableop_resource:	
identityขBiasAdd/ReadVariableOpขMatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????Q
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:?????????b
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:?????????w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:?????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:?????????
 
_user_specified_nameinputs
๚	
c
D__inference_dropout_5_layer_call_and_return_conditional_losses_91809

inputs
identityR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @e
dropout/MulMulinputsdropout/Const:output:0*
T0*(
_output_shapes
:?????????C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:?????????*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?ง
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:?????????p
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:?????????j
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*(
_output_shapes
:?????????Z
IdentityIdentitydropout/Mul_1:z:0*
T0*(
_output_shapes
:?????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:?????????:P L
(
_output_shapes
:?????????
 
_user_specified_nameinputs
	
ี
6__inference_batch_normalization_11_layer_call_fn_93103

inputs
unknown:	
	unknown_0:	
	unknown_1:	
	unknown_2:	
identityขStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,???????????????????????????*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Z
fURS
Q__inference_batch_normalization_11_layer_call_and_return_conditional_losses_91222
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*B
_output_shapes0
.:,???????????????????????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,???????????????????????????: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,???????????????????????????
 
_user_specified_nameinputs

f
J__inference_max_pooling2d_7_layer_call_and_return_conditional_losses_93162

inputs
identityข
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4????????????????????????????????????*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
ข

๔
B__inference_dense_8_layer_call_and_return_conditional_losses_93543

inputs1
matmul_readvariableop_resource:	-
biasadd_readvariableop_resource:
identityขBiasAdd/ReadVariableOpขMatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????V
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:?????????`
IdentityIdentitySoftmax:softmax:0^NoOp*
T0*'
_output_shapes
:?????????w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:?????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:?????????
 
_user_specified_nameinputs


D__inference_conv2d_14_layer_call_and_return_conditional_losses_93346

inputs:
conv2d_readvariableop_resource:.
biasadd_readvariableop_resource:	
identityขBiasAdd/ReadVariableOpขConv2D/ReadVariableOp~
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:*
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????*
paddingSAME*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????Y
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:?????????j
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:?????????w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :?????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:?????????
 
_user_specified_nameinputs
ฑ
E
)__inference_flatten_2_layer_call_fn_93423

inputs
identityฐ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_flatten_2_layer_call_and_return_conditional_losses_91636a
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:?????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:?????????:X T
0
_output_shapes
:?????????
 
_user_specified_nameinputs
ต
์
#__inference_signature_wrapper_92515
rescaling_2_input!
unknown:`
	unknown_0:`
	unknown_1:`
	unknown_2:`
	unknown_3:`
	unknown_4:`$
	unknown_5:`
	unknown_6:	
	unknown_7:	
	unknown_8:	
	unknown_9:	

unknown_10:	&

unknown_11:

unknown_12:	

unknown_13:	

unknown_14:	

unknown_15:	

unknown_16:	&

unknown_17:

unknown_18:	

unknown_19:	

unknown_20:	

unknown_21:	

unknown_22:	&

unknown_23:

unknown_24:	

unknown_25:	

unknown_26:	

unknown_27:	

unknown_28:	

unknown_29:


unknown_30:	

unknown_31:


unknown_32:	

unknown_33:	

unknown_34:
identityขStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallrescaling_2_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34*0
Tin)
'2%*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*F
_read_only_resource_inputs(
&$	
 !"#$*-
config_proto

CPU

GPU 2J 8 *)
f$R"
 __inference__wrapped_model_91124o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*x
_input_shapesg
e:?????????เ: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:d `
1
_output_shapes
:?????????เ
+
_user_specified_namerescaling_2_input
?
b
D__inference_dropout_4_layer_call_and_return_conditional_losses_91660

inputs

identity_1O
IdentityIdentityinputs*
T0*(
_output_shapes
:?????????\

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:?????????"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:?????????:P L
(
_output_shapes
:?????????
 
_user_specified_nameinputs
งั
่&
 __inference__wrapped_model_91124
rescaling_2_inputO
5sequential_2_conv2d_10_conv2d_readvariableop_resource:`D
6sequential_2_conv2d_10_biasadd_readvariableop_resource:`I
;sequential_2_batch_normalization_10_readvariableop_resource:`K
=sequential_2_batch_normalization_10_readvariableop_1_resource:`Z
Lsequential_2_batch_normalization_10_fusedbatchnormv3_readvariableop_resource:`\
Nsequential_2_batch_normalization_10_fusedbatchnormv3_readvariableop_1_resource:`P
5sequential_2_conv2d_11_conv2d_readvariableop_resource:`E
6sequential_2_conv2d_11_biasadd_readvariableop_resource:	J
;sequential_2_batch_normalization_11_readvariableop_resource:	L
=sequential_2_batch_normalization_11_readvariableop_1_resource:	[
Lsequential_2_batch_normalization_11_fusedbatchnormv3_readvariableop_resource:	]
Nsequential_2_batch_normalization_11_fusedbatchnormv3_readvariableop_1_resource:	Q
5sequential_2_conv2d_12_conv2d_readvariableop_resource:E
6sequential_2_conv2d_12_biasadd_readvariableop_resource:	J
;sequential_2_batch_normalization_12_readvariableop_resource:	L
=sequential_2_batch_normalization_12_readvariableop_1_resource:	[
Lsequential_2_batch_normalization_12_fusedbatchnormv3_readvariableop_resource:	]
Nsequential_2_batch_normalization_12_fusedbatchnormv3_readvariableop_1_resource:	Q
5sequential_2_conv2d_13_conv2d_readvariableop_resource:E
6sequential_2_conv2d_13_biasadd_readvariableop_resource:	J
;sequential_2_batch_normalization_13_readvariableop_resource:	L
=sequential_2_batch_normalization_13_readvariableop_1_resource:	[
Lsequential_2_batch_normalization_13_fusedbatchnormv3_readvariableop_resource:	]
Nsequential_2_batch_normalization_13_fusedbatchnormv3_readvariableop_1_resource:	Q
5sequential_2_conv2d_14_conv2d_readvariableop_resource:E
6sequential_2_conv2d_14_biasadd_readvariableop_resource:	J
;sequential_2_batch_normalization_14_readvariableop_resource:	L
=sequential_2_batch_normalization_14_readvariableop_1_resource:	[
Lsequential_2_batch_normalization_14_fusedbatchnormv3_readvariableop_resource:	]
Nsequential_2_batch_normalization_14_fusedbatchnormv3_readvariableop_1_resource:	G
3sequential_2_dense_6_matmul_readvariableop_resource:
C
4sequential_2_dense_6_biasadd_readvariableop_resource:	G
3sequential_2_dense_7_matmul_readvariableop_resource:
C
4sequential_2_dense_7_biasadd_readvariableop_resource:	F
3sequential_2_dense_8_matmul_readvariableop_resource:	B
4sequential_2_dense_8_biasadd_readvariableop_resource:
identityขCsequential_2/batch_normalization_10/FusedBatchNormV3/ReadVariableOpขEsequential_2/batch_normalization_10/FusedBatchNormV3/ReadVariableOp_1ข2sequential_2/batch_normalization_10/ReadVariableOpข4sequential_2/batch_normalization_10/ReadVariableOp_1ขCsequential_2/batch_normalization_11/FusedBatchNormV3/ReadVariableOpขEsequential_2/batch_normalization_11/FusedBatchNormV3/ReadVariableOp_1ข2sequential_2/batch_normalization_11/ReadVariableOpข4sequential_2/batch_normalization_11/ReadVariableOp_1ขCsequential_2/batch_normalization_12/FusedBatchNormV3/ReadVariableOpขEsequential_2/batch_normalization_12/FusedBatchNormV3/ReadVariableOp_1ข2sequential_2/batch_normalization_12/ReadVariableOpข4sequential_2/batch_normalization_12/ReadVariableOp_1ขCsequential_2/batch_normalization_13/FusedBatchNormV3/ReadVariableOpขEsequential_2/batch_normalization_13/FusedBatchNormV3/ReadVariableOp_1ข2sequential_2/batch_normalization_13/ReadVariableOpข4sequential_2/batch_normalization_13/ReadVariableOp_1ขCsequential_2/batch_normalization_14/FusedBatchNormV3/ReadVariableOpขEsequential_2/batch_normalization_14/FusedBatchNormV3/ReadVariableOp_1ข2sequential_2/batch_normalization_14/ReadVariableOpข4sequential_2/batch_normalization_14/ReadVariableOp_1ข-sequential_2/conv2d_10/BiasAdd/ReadVariableOpข,sequential_2/conv2d_10/Conv2D/ReadVariableOpข-sequential_2/conv2d_11/BiasAdd/ReadVariableOpข,sequential_2/conv2d_11/Conv2D/ReadVariableOpข-sequential_2/conv2d_12/BiasAdd/ReadVariableOpข,sequential_2/conv2d_12/Conv2D/ReadVariableOpข-sequential_2/conv2d_13/BiasAdd/ReadVariableOpข,sequential_2/conv2d_13/Conv2D/ReadVariableOpข-sequential_2/conv2d_14/BiasAdd/ReadVariableOpข,sequential_2/conv2d_14/Conv2D/ReadVariableOpข+sequential_2/dense_6/BiasAdd/ReadVariableOpข*sequential_2/dense_6/MatMul/ReadVariableOpข+sequential_2/dense_7/BiasAdd/ReadVariableOpข*sequential_2/dense_7/MatMul/ReadVariableOpข+sequential_2/dense_8/BiasAdd/ReadVariableOpข*sequential_2/dense_8/MatMul/ReadVariableOpd
sequential_2/rescaling_2/Cast/xConst*
_output_shapes
: *
dtype0*
valueB
 *;f
!sequential_2/rescaling_2/Cast_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *    
sequential_2/rescaling_2/mulMulrescaling_2_input(sequential_2/rescaling_2/Cast/x:output:0*
T0*1
_output_shapes
:?????????เฏ
sequential_2/rescaling_2/addAddV2 sequential_2/rescaling_2/mul:z:0*sequential_2/rescaling_2/Cast_1/x:output:0*
T0*1
_output_shapes
:?????????เช
,sequential_2/conv2d_10/Conv2D/ReadVariableOpReadVariableOp5sequential_2_conv2d_10_conv2d_readvariableop_resource*&
_output_shapes
:`*
dtype0ใ
sequential_2/conv2d_10/Conv2DConv2D sequential_2/rescaling_2/add:z:04sequential_2/conv2d_10/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????v`*
paddingVALID*
strides
?
-sequential_2/conv2d_10/BiasAdd/ReadVariableOpReadVariableOp6sequential_2_conv2d_10_biasadd_readvariableop_resource*
_output_shapes
:`*
dtype0ร
sequential_2/conv2d_10/BiasAddBiasAdd&sequential_2/conv2d_10/Conv2D:output:05sequential_2/conv2d_10/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????v`
sequential_2/conv2d_10/ReluRelu'sequential_2/conv2d_10/BiasAdd:output:0*
T0*0
_output_shapes
:?????????v`ช
2sequential_2/batch_normalization_10/ReadVariableOpReadVariableOp;sequential_2_batch_normalization_10_readvariableop_resource*
_output_shapes
:`*
dtype0ฎ
4sequential_2/batch_normalization_10/ReadVariableOp_1ReadVariableOp=sequential_2_batch_normalization_10_readvariableop_1_resource*
_output_shapes
:`*
dtype0ฬ
Csequential_2/batch_normalization_10/FusedBatchNormV3/ReadVariableOpReadVariableOpLsequential_2_batch_normalization_10_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:`*
dtype0ะ
Esequential_2/batch_normalization_10/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpNsequential_2_batch_normalization_10_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:`*
dtype0
4sequential_2/batch_normalization_10/FusedBatchNormV3FusedBatchNormV3)sequential_2/conv2d_10/Relu:activations:0:sequential_2/batch_normalization_10/ReadVariableOp:value:0<sequential_2/batch_normalization_10/ReadVariableOp_1:value:0Ksequential_2/batch_normalization_10/FusedBatchNormV3/ReadVariableOp:value:0Msequential_2/batch_normalization_10/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*L
_output_shapes:
8:?????????v`:`:`:`:`:*
epsilon%o:*
is_training( ึ
$sequential_2/max_pooling2d_6/MaxPoolMaxPool8sequential_2/batch_normalization_10/FusedBatchNormV3:y:0*/
_output_shapes
:?????????:N`*
ksize
*
paddingVALID*
strides
ซ
,sequential_2/conv2d_11/Conv2D/ReadVariableOpReadVariableOp5sequential_2_conv2d_11_conv2d_readvariableop_resource*'
_output_shapes
:`*
dtype0๏
sequential_2/conv2d_11/Conv2DConv2D-sequential_2/max_pooling2d_6/MaxPool:output:04sequential_2/conv2d_11/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????*
paddingSAME*
strides
ก
-sequential_2/conv2d_11/BiasAdd/ReadVariableOpReadVariableOp6sequential_2_conv2d_11_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0ร
sequential_2/conv2d_11/BiasAddBiasAdd&sequential_2/conv2d_11/Conv2D:output:05sequential_2/conv2d_11/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????
sequential_2/conv2d_11/ReluRelu'sequential_2/conv2d_11/BiasAdd:output:0*
T0*0
_output_shapes
:?????????ซ
2sequential_2/batch_normalization_11/ReadVariableOpReadVariableOp;sequential_2_batch_normalization_11_readvariableop_resource*
_output_shapes	
:*
dtype0ฏ
4sequential_2/batch_normalization_11/ReadVariableOp_1ReadVariableOp=sequential_2_batch_normalization_11_readvariableop_1_resource*
_output_shapes	
:*
dtype0อ
Csequential_2/batch_normalization_11/FusedBatchNormV3/ReadVariableOpReadVariableOpLsequential_2_batch_normalization_11_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype0ั
Esequential_2/batch_normalization_11/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpNsequential_2_batch_normalization_11_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype0
4sequential_2/batch_normalization_11/FusedBatchNormV3FusedBatchNormV3)sequential_2/conv2d_11/Relu:activations:0:sequential_2/batch_normalization_11/ReadVariableOp:value:0<sequential_2/batch_normalization_11/ReadVariableOp_1:value:0Ksequential_2/batch_normalization_11/FusedBatchNormV3/ReadVariableOp:value:0Msequential_2/batch_normalization_11/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:?????????:::::*
epsilon%o:*
is_training( ื
$sequential_2/max_pooling2d_7/MaxPoolMaxPool8sequential_2/batch_normalization_11/FusedBatchNormV3:y:0*0
_output_shapes
:?????????	*
ksize
*
paddingVALID*
strides
ฌ
,sequential_2/conv2d_12/Conv2D/ReadVariableOpReadVariableOp5sequential_2_conv2d_12_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0๏
sequential_2/conv2d_12/Conv2DConv2D-sequential_2/max_pooling2d_7/MaxPool:output:04sequential_2/conv2d_12/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????*
paddingSAME*
strides
ก
-sequential_2/conv2d_12/BiasAdd/ReadVariableOpReadVariableOp6sequential_2_conv2d_12_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0ร
sequential_2/conv2d_12/BiasAddBiasAdd&sequential_2/conv2d_12/Conv2D:output:05sequential_2/conv2d_12/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????
sequential_2/conv2d_12/ReluRelu'sequential_2/conv2d_12/BiasAdd:output:0*
T0*0
_output_shapes
:?????????ซ
2sequential_2/batch_normalization_12/ReadVariableOpReadVariableOp;sequential_2_batch_normalization_12_readvariableop_resource*
_output_shapes	
:*
dtype0ฏ
4sequential_2/batch_normalization_12/ReadVariableOp_1ReadVariableOp=sequential_2_batch_normalization_12_readvariableop_1_resource*
_output_shapes	
:*
dtype0อ
Csequential_2/batch_normalization_12/FusedBatchNormV3/ReadVariableOpReadVariableOpLsequential_2_batch_normalization_12_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype0ั
Esequential_2/batch_normalization_12/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpNsequential_2_batch_normalization_12_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype0
4sequential_2/batch_normalization_12/FusedBatchNormV3FusedBatchNormV3)sequential_2/conv2d_12/Relu:activations:0:sequential_2/batch_normalization_12/ReadVariableOp:value:0<sequential_2/batch_normalization_12/ReadVariableOp_1:value:0Ksequential_2/batch_normalization_12/FusedBatchNormV3/ReadVariableOp:value:0Msequential_2/batch_normalization_12/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:?????????:::::*
epsilon%o:*
is_training( ฌ
,sequential_2/conv2d_13/Conv2D/ReadVariableOpReadVariableOp5sequential_2_conv2d_13_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0๚
sequential_2/conv2d_13/Conv2DConv2D8sequential_2/batch_normalization_12/FusedBatchNormV3:y:04sequential_2/conv2d_13/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????*
paddingSAME*
strides
ก
-sequential_2/conv2d_13/BiasAdd/ReadVariableOpReadVariableOp6sequential_2_conv2d_13_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0ร
sequential_2/conv2d_13/BiasAddBiasAdd&sequential_2/conv2d_13/Conv2D:output:05sequential_2/conv2d_13/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????
sequential_2/conv2d_13/ReluRelu'sequential_2/conv2d_13/BiasAdd:output:0*
T0*0
_output_shapes
:?????????ซ
2sequential_2/batch_normalization_13/ReadVariableOpReadVariableOp;sequential_2_batch_normalization_13_readvariableop_resource*
_output_shapes	
:*
dtype0ฏ
4sequential_2/batch_normalization_13/ReadVariableOp_1ReadVariableOp=sequential_2_batch_normalization_13_readvariableop_1_resource*
_output_shapes	
:*
dtype0อ
Csequential_2/batch_normalization_13/FusedBatchNormV3/ReadVariableOpReadVariableOpLsequential_2_batch_normalization_13_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype0ั
Esequential_2/batch_normalization_13/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpNsequential_2_batch_normalization_13_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype0
4sequential_2/batch_normalization_13/FusedBatchNormV3FusedBatchNormV3)sequential_2/conv2d_13/Relu:activations:0:sequential_2/batch_normalization_13/ReadVariableOp:value:0<sequential_2/batch_normalization_13/ReadVariableOp_1:value:0Ksequential_2/batch_normalization_13/FusedBatchNormV3/ReadVariableOp:value:0Msequential_2/batch_normalization_13/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:?????????:::::*
epsilon%o:*
is_training( ฌ
,sequential_2/conv2d_14/Conv2D/ReadVariableOpReadVariableOp5sequential_2_conv2d_14_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0๚
sequential_2/conv2d_14/Conv2DConv2D8sequential_2/batch_normalization_13/FusedBatchNormV3:y:04sequential_2/conv2d_14/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????*
paddingSAME*
strides
ก
-sequential_2/conv2d_14/BiasAdd/ReadVariableOpReadVariableOp6sequential_2_conv2d_14_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0ร
sequential_2/conv2d_14/BiasAddBiasAdd&sequential_2/conv2d_14/Conv2D:output:05sequential_2/conv2d_14/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????
sequential_2/conv2d_14/ReluRelu'sequential_2/conv2d_14/BiasAdd:output:0*
T0*0
_output_shapes
:?????????ซ
2sequential_2/batch_normalization_14/ReadVariableOpReadVariableOp;sequential_2_batch_normalization_14_readvariableop_resource*
_output_shapes	
:*
dtype0ฏ
4sequential_2/batch_normalization_14/ReadVariableOp_1ReadVariableOp=sequential_2_batch_normalization_14_readvariableop_1_resource*
_output_shapes	
:*
dtype0อ
Csequential_2/batch_normalization_14/FusedBatchNormV3/ReadVariableOpReadVariableOpLsequential_2_batch_normalization_14_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype0ั
Esequential_2/batch_normalization_14/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpNsequential_2_batch_normalization_14_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype0
4sequential_2/batch_normalization_14/FusedBatchNormV3FusedBatchNormV3)sequential_2/conv2d_14/Relu:activations:0:sequential_2/batch_normalization_14/ReadVariableOp:value:0<sequential_2/batch_normalization_14/ReadVariableOp_1:value:0Ksequential_2/batch_normalization_14/FusedBatchNormV3/ReadVariableOp:value:0Msequential_2/batch_normalization_14/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:?????????:::::*
epsilon%o:*
is_training( ื
$sequential_2/max_pooling2d_8/MaxPoolMaxPool8sequential_2/batch_normalization_14/FusedBatchNormV3:y:0*0
_output_shapes
:?????????*
ksize
*
paddingVALID*
strides
m
sequential_2/flatten_2/ConstConst*
_output_shapes
:*
dtype0*
valueB"????   ฒ
sequential_2/flatten_2/ReshapeReshape-sequential_2/max_pooling2d_8/MaxPool:output:0%sequential_2/flatten_2/Const:output:0*
T0*(
_output_shapes
:??????????
*sequential_2/dense_6/MatMul/ReadVariableOpReadVariableOp3sequential_2_dense_6_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype0ต
sequential_2/dense_6/MatMulMatMul'sequential_2/flatten_2/Reshape:output:02sequential_2/dense_6/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????
+sequential_2/dense_6/BiasAdd/ReadVariableOpReadVariableOp4sequential_2_dense_6_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0ถ
sequential_2/dense_6/BiasAddBiasAdd%sequential_2/dense_6/MatMul:product:03sequential_2/dense_6/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????{
sequential_2/dense_6/ReluRelu%sequential_2/dense_6/BiasAdd:output:0*
T0*(
_output_shapes
:?????????
sequential_2/dropout_4/IdentityIdentity'sequential_2/dense_6/Relu:activations:0*
T0*(
_output_shapes
:??????????
*sequential_2/dense_7/MatMul/ReadVariableOpReadVariableOp3sequential_2_dense_7_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype0ถ
sequential_2/dense_7/MatMulMatMul(sequential_2/dropout_4/Identity:output:02sequential_2/dense_7/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????
+sequential_2/dense_7/BiasAdd/ReadVariableOpReadVariableOp4sequential_2_dense_7_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0ถ
sequential_2/dense_7/BiasAddBiasAdd%sequential_2/dense_7/MatMul:product:03sequential_2/dense_7/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????{
sequential_2/dense_7/ReluRelu%sequential_2/dense_7/BiasAdd:output:0*
T0*(
_output_shapes
:?????????
sequential_2/dropout_5/IdentityIdentity'sequential_2/dense_7/Relu:activations:0*
T0*(
_output_shapes
:?????????
*sequential_2/dense_8/MatMul/ReadVariableOpReadVariableOp3sequential_2_dense_8_matmul_readvariableop_resource*
_output_shapes
:	*
dtype0ต
sequential_2/dense_8/MatMulMatMul(sequential_2/dropout_5/Identity:output:02sequential_2/dense_8/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
+sequential_2/dense_8/BiasAdd/ReadVariableOpReadVariableOp4sequential_2_dense_8_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0ต
sequential_2/dense_8/BiasAddBiasAdd%sequential_2/dense_8/MatMul:product:03sequential_2/dense_8/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
sequential_2/dense_8/SoftmaxSoftmax%sequential_2/dense_8/BiasAdd:output:0*
T0*'
_output_shapes
:?????????u
IdentityIdentity&sequential_2/dense_8/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:?????????
NoOpNoOpD^sequential_2/batch_normalization_10/FusedBatchNormV3/ReadVariableOpF^sequential_2/batch_normalization_10/FusedBatchNormV3/ReadVariableOp_13^sequential_2/batch_normalization_10/ReadVariableOp5^sequential_2/batch_normalization_10/ReadVariableOp_1D^sequential_2/batch_normalization_11/FusedBatchNormV3/ReadVariableOpF^sequential_2/batch_normalization_11/FusedBatchNormV3/ReadVariableOp_13^sequential_2/batch_normalization_11/ReadVariableOp5^sequential_2/batch_normalization_11/ReadVariableOp_1D^sequential_2/batch_normalization_12/FusedBatchNormV3/ReadVariableOpF^sequential_2/batch_normalization_12/FusedBatchNormV3/ReadVariableOp_13^sequential_2/batch_normalization_12/ReadVariableOp5^sequential_2/batch_normalization_12/ReadVariableOp_1D^sequential_2/batch_normalization_13/FusedBatchNormV3/ReadVariableOpF^sequential_2/batch_normalization_13/FusedBatchNormV3/ReadVariableOp_13^sequential_2/batch_normalization_13/ReadVariableOp5^sequential_2/batch_normalization_13/ReadVariableOp_1D^sequential_2/batch_normalization_14/FusedBatchNormV3/ReadVariableOpF^sequential_2/batch_normalization_14/FusedBatchNormV3/ReadVariableOp_13^sequential_2/batch_normalization_14/ReadVariableOp5^sequential_2/batch_normalization_14/ReadVariableOp_1.^sequential_2/conv2d_10/BiasAdd/ReadVariableOp-^sequential_2/conv2d_10/Conv2D/ReadVariableOp.^sequential_2/conv2d_11/BiasAdd/ReadVariableOp-^sequential_2/conv2d_11/Conv2D/ReadVariableOp.^sequential_2/conv2d_12/BiasAdd/ReadVariableOp-^sequential_2/conv2d_12/Conv2D/ReadVariableOp.^sequential_2/conv2d_13/BiasAdd/ReadVariableOp-^sequential_2/conv2d_13/Conv2D/ReadVariableOp.^sequential_2/conv2d_14/BiasAdd/ReadVariableOp-^sequential_2/conv2d_14/Conv2D/ReadVariableOp,^sequential_2/dense_6/BiasAdd/ReadVariableOp+^sequential_2/dense_6/MatMul/ReadVariableOp,^sequential_2/dense_7/BiasAdd/ReadVariableOp+^sequential_2/dense_7/MatMul/ReadVariableOp,^sequential_2/dense_8/BiasAdd/ReadVariableOp+^sequential_2/dense_8/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*x
_input_shapesg
e:?????????เ: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2
Csequential_2/batch_normalization_10/FusedBatchNormV3/ReadVariableOpCsequential_2/batch_normalization_10/FusedBatchNormV3/ReadVariableOp2
Esequential_2/batch_normalization_10/FusedBatchNormV3/ReadVariableOp_1Esequential_2/batch_normalization_10/FusedBatchNormV3/ReadVariableOp_12h
2sequential_2/batch_normalization_10/ReadVariableOp2sequential_2/batch_normalization_10/ReadVariableOp2l
4sequential_2/batch_normalization_10/ReadVariableOp_14sequential_2/batch_normalization_10/ReadVariableOp_12
Csequential_2/batch_normalization_11/FusedBatchNormV3/ReadVariableOpCsequential_2/batch_normalization_11/FusedBatchNormV3/ReadVariableOp2
Esequential_2/batch_normalization_11/FusedBatchNormV3/ReadVariableOp_1Esequential_2/batch_normalization_11/FusedBatchNormV3/ReadVariableOp_12h
2sequential_2/batch_normalization_11/ReadVariableOp2sequential_2/batch_normalization_11/ReadVariableOp2l
4sequential_2/batch_normalization_11/ReadVariableOp_14sequential_2/batch_normalization_11/ReadVariableOp_12
Csequential_2/batch_normalization_12/FusedBatchNormV3/ReadVariableOpCsequential_2/batch_normalization_12/FusedBatchNormV3/ReadVariableOp2
Esequential_2/batch_normalization_12/FusedBatchNormV3/ReadVariableOp_1Esequential_2/batch_normalization_12/FusedBatchNormV3/ReadVariableOp_12h
2sequential_2/batch_normalization_12/ReadVariableOp2sequential_2/batch_normalization_12/ReadVariableOp2l
4sequential_2/batch_normalization_12/ReadVariableOp_14sequential_2/batch_normalization_12/ReadVariableOp_12
Csequential_2/batch_normalization_13/FusedBatchNormV3/ReadVariableOpCsequential_2/batch_normalization_13/FusedBatchNormV3/ReadVariableOp2
Esequential_2/batch_normalization_13/FusedBatchNormV3/ReadVariableOp_1Esequential_2/batch_normalization_13/FusedBatchNormV3/ReadVariableOp_12h
2sequential_2/batch_normalization_13/ReadVariableOp2sequential_2/batch_normalization_13/ReadVariableOp2l
4sequential_2/batch_normalization_13/ReadVariableOp_14sequential_2/batch_normalization_13/ReadVariableOp_12
Csequential_2/batch_normalization_14/FusedBatchNormV3/ReadVariableOpCsequential_2/batch_normalization_14/FusedBatchNormV3/ReadVariableOp2
Esequential_2/batch_normalization_14/FusedBatchNormV3/ReadVariableOp_1Esequential_2/batch_normalization_14/FusedBatchNormV3/ReadVariableOp_12h
2sequential_2/batch_normalization_14/ReadVariableOp2sequential_2/batch_normalization_14/ReadVariableOp2l
4sequential_2/batch_normalization_14/ReadVariableOp_14sequential_2/batch_normalization_14/ReadVariableOp_12^
-sequential_2/conv2d_10/BiasAdd/ReadVariableOp-sequential_2/conv2d_10/BiasAdd/ReadVariableOp2\
,sequential_2/conv2d_10/Conv2D/ReadVariableOp,sequential_2/conv2d_10/Conv2D/ReadVariableOp2^
-sequential_2/conv2d_11/BiasAdd/ReadVariableOp-sequential_2/conv2d_11/BiasAdd/ReadVariableOp2\
,sequential_2/conv2d_11/Conv2D/ReadVariableOp,sequential_2/conv2d_11/Conv2D/ReadVariableOp2^
-sequential_2/conv2d_12/BiasAdd/ReadVariableOp-sequential_2/conv2d_12/BiasAdd/ReadVariableOp2\
,sequential_2/conv2d_12/Conv2D/ReadVariableOp,sequential_2/conv2d_12/Conv2D/ReadVariableOp2^
-sequential_2/conv2d_13/BiasAdd/ReadVariableOp-sequential_2/conv2d_13/BiasAdd/ReadVariableOp2\
,sequential_2/conv2d_13/Conv2D/ReadVariableOp,sequential_2/conv2d_13/Conv2D/ReadVariableOp2^
-sequential_2/conv2d_14/BiasAdd/ReadVariableOp-sequential_2/conv2d_14/BiasAdd/ReadVariableOp2\
,sequential_2/conv2d_14/Conv2D/ReadVariableOp,sequential_2/conv2d_14/Conv2D/ReadVariableOp2Z
+sequential_2/dense_6/BiasAdd/ReadVariableOp+sequential_2/dense_6/BiasAdd/ReadVariableOp2X
*sequential_2/dense_6/MatMul/ReadVariableOp*sequential_2/dense_6/MatMul/ReadVariableOp2Z
+sequential_2/dense_7/BiasAdd/ReadVariableOp+sequential_2/dense_7/BiasAdd/ReadVariableOp2X
*sequential_2/dense_7/MatMul/ReadVariableOp*sequential_2/dense_7/MatMul/ReadVariableOp2Z
+sequential_2/dense_8/BiasAdd/ReadVariableOp+sequential_2/dense_8/BiasAdd/ReadVariableOp2X
*sequential_2/dense_8/MatMul/ReadVariableOp*sequential_2/dense_8/MatMul/ReadVariableOp:d `
1
_output_shapes
:?????????เ
+
_user_specified_namerescaling_2_input

ฤ
Q__inference_batch_normalization_14_layer_call_and_return_conditional_losses_91457

inputs&
readvariableop_resource:	(
readvariableop_1_resource:	7
(fusedbatchnormv3_readvariableop_resource:	9
*fusedbatchnormv3_readvariableop_1_resource:	
identityขAssignNewValueขAssignNewValue_1ขFusedBatchNormV3/ReadVariableOpข!FusedBatchNormV3/ReadVariableOp_1ขReadVariableOpขReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:*
dtype0
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype0
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype0?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,???????????????????????????:::::*
epsilon%o:*
exponential_avg_factor%
ื#<ฦ
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(ะ
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(~
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,???????????????????????????ิ
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,???????????????????????????: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,???????????????????????????
 
_user_specified_nameinputs
?
?
Q__inference_batch_normalization_14_layer_call_and_return_conditional_losses_91426

inputs&
readvariableop_resource:	(
readvariableop_1_resource:	7
(fusedbatchnormv3_readvariableop_resource:	9
*fusedbatchnormv3_readvariableop_1_resource:	
identityขFusedBatchNormV3/ReadVariableOpข!FusedBatchNormV3/ReadVariableOp_1ขReadVariableOpขReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:*
dtype0
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype0
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype0อ
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,???????????????????????????:::::*
epsilon%o:*
is_training( ~
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,???????????????????????????ฐ
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,???????????????????????????: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,???????????????????????????
 
_user_specified_nameinputs
๒e

G__inference_sequential_2_layer_call_and_return_conditional_losses_92436
rescaling_2_input)
conv2d_10_92344:`
conv2d_10_92346:`*
batch_normalization_10_92349:`*
batch_normalization_10_92351:`*
batch_normalization_10_92353:`*
batch_normalization_10_92355:`*
conv2d_11_92359:`
conv2d_11_92361:	+
batch_normalization_11_92364:	+
batch_normalization_11_92366:	+
batch_normalization_11_92368:	+
batch_normalization_11_92370:	+
conv2d_12_92374:
conv2d_12_92376:	+
batch_normalization_12_92379:	+
batch_normalization_12_92381:	+
batch_normalization_12_92383:	+
batch_normalization_12_92385:	+
conv2d_13_92388:
conv2d_13_92390:	+
batch_normalization_13_92393:	+
batch_normalization_13_92395:	+
batch_normalization_13_92397:	+
batch_normalization_13_92399:	+
conv2d_14_92402:
conv2d_14_92404:	+
batch_normalization_14_92407:	+
batch_normalization_14_92409:	+
batch_normalization_14_92411:	+
batch_normalization_14_92413:	!
dense_6_92418:

dense_6_92420:	!
dense_7_92424:

dense_7_92426:	 
dense_8_92430:	
dense_8_92432:
identityข.batch_normalization_10/StatefulPartitionedCallข.batch_normalization_11/StatefulPartitionedCallข.batch_normalization_12/StatefulPartitionedCallข.batch_normalization_13/StatefulPartitionedCallข.batch_normalization_14/StatefulPartitionedCallข!conv2d_10/StatefulPartitionedCallข!conv2d_11/StatefulPartitionedCallข!conv2d_12/StatefulPartitionedCallข!conv2d_13/StatefulPartitionedCallข!conv2d_14/StatefulPartitionedCallขdense_6/StatefulPartitionedCallขdense_7/StatefulPartitionedCallขdense_8/StatefulPartitionedCallข!dropout_4/StatefulPartitionedCallข!dropout_5/StatefulPartitionedCallา
rescaling_2/PartitionedCallPartitionedCallrescaling_2_input*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:?????????เ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_rescaling_2_layer_call_and_return_conditional_losses_91495
!conv2d_10/StatefulPartitionedCallStatefulPartitionedCall$rescaling_2/PartitionedCall:output:0conv2d_10_92344conv2d_10_92346*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????v`*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_conv2d_10_layer_call_and_return_conditional_losses_91508
.batch_normalization_10/StatefulPartitionedCallStatefulPartitionedCall*conv2d_10/StatefulPartitionedCall:output:0batch_normalization_10_92349batch_normalization_10_92351batch_normalization_10_92353batch_normalization_10_92355*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????v`*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Z
fURS
Q__inference_batch_normalization_10_layer_call_and_return_conditional_losses_91177?
max_pooling2d_6/PartitionedCallPartitionedCall7batch_normalization_10/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????:N`* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *S
fNRL
J__inference_max_pooling2d_6_layer_call_and_return_conditional_losses_91197
!conv2d_11/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_6/PartitionedCall:output:0conv2d_11_92359conv2d_11_92361*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_conv2d_11_layer_call_and_return_conditional_losses_91535
.batch_normalization_11/StatefulPartitionedCallStatefulPartitionedCall*conv2d_11/StatefulPartitionedCall:output:0batch_normalization_11_92364batch_normalization_11_92366batch_normalization_11_92368batch_normalization_11_92370*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Z
fURS
Q__inference_batch_normalization_11_layer_call_and_return_conditional_losses_91253?
max_pooling2d_7/PartitionedCallPartitionedCall7batch_normalization_11/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????	* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *S
fNRL
J__inference_max_pooling2d_7_layer_call_and_return_conditional_losses_91273
!conv2d_12/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_7/PartitionedCall:output:0conv2d_12_92374conv2d_12_92376*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_conv2d_12_layer_call_and_return_conditional_losses_91562
.batch_normalization_12/StatefulPartitionedCallStatefulPartitionedCall*conv2d_12/StatefulPartitionedCall:output:0batch_normalization_12_92379batch_normalization_12_92381batch_normalization_12_92383batch_normalization_12_92385*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Z
fURS
Q__inference_batch_normalization_12_layer_call_and_return_conditional_losses_91329ซ
!conv2d_13/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_12/StatefulPartitionedCall:output:0conv2d_13_92388conv2d_13_92390*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_conv2d_13_layer_call_and_return_conditional_losses_91588
.batch_normalization_13/StatefulPartitionedCallStatefulPartitionedCall*conv2d_13/StatefulPartitionedCall:output:0batch_normalization_13_92393batch_normalization_13_92395batch_normalization_13_92397batch_normalization_13_92399*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Z
fURS
Q__inference_batch_normalization_13_layer_call_and_return_conditional_losses_91393ซ
!conv2d_14/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_13/StatefulPartitionedCall:output:0conv2d_14_92402conv2d_14_92404*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_conv2d_14_layer_call_and_return_conditional_losses_91614
.batch_normalization_14/StatefulPartitionedCallStatefulPartitionedCall*conv2d_14/StatefulPartitionedCall:output:0batch_normalization_14_92407batch_normalization_14_92409batch_normalization_14_92411batch_normalization_14_92413*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Z
fURS
Q__inference_batch_normalization_14_layer_call_and_return_conditional_losses_91457?
max_pooling2d_8/PartitionedCallPartitionedCall7batch_normalization_14/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *S
fNRL
J__inference_max_pooling2d_8_layer_call_and_return_conditional_losses_91477?
flatten_2/PartitionedCallPartitionedCall(max_pooling2d_8/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_flatten_2_layer_call_and_return_conditional_losses_91636
dense_6/StatefulPartitionedCallStatefulPartitionedCall"flatten_2/PartitionedCall:output:0dense_6_92418dense_6_92420*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_dense_6_layer_call_and_return_conditional_losses_91649์
!dropout_4/StatefulPartitionedCallStatefulPartitionedCall(dense_6/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dropout_4_layer_call_and_return_conditional_losses_91842
dense_7/StatefulPartitionedCallStatefulPartitionedCall*dropout_4/StatefulPartitionedCall:output:0dense_7_92424dense_7_92426*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_dense_7_layer_call_and_return_conditional_losses_91673
!dropout_5/StatefulPartitionedCallStatefulPartitionedCall(dense_7/StatefulPartitionedCall:output:0"^dropout_4/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dropout_5_layer_call_and_return_conditional_losses_91809
dense_8/StatefulPartitionedCallStatefulPartitionedCall*dropout_5/StatefulPartitionedCall:output:0dense_8_92430dense_8_92432*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_dense_8_layer_call_and_return_conditional_losses_91697w
IdentityIdentity(dense_8/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????
NoOpNoOp/^batch_normalization_10/StatefulPartitionedCall/^batch_normalization_11/StatefulPartitionedCall/^batch_normalization_12/StatefulPartitionedCall/^batch_normalization_13/StatefulPartitionedCall/^batch_normalization_14/StatefulPartitionedCall"^conv2d_10/StatefulPartitionedCall"^conv2d_11/StatefulPartitionedCall"^conv2d_12/StatefulPartitionedCall"^conv2d_13/StatefulPartitionedCall"^conv2d_14/StatefulPartitionedCall ^dense_6/StatefulPartitionedCall ^dense_7/StatefulPartitionedCall ^dense_8/StatefulPartitionedCall"^dropout_4/StatefulPartitionedCall"^dropout_5/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*x
_input_shapesg
e:?????????เ: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2`
.batch_normalization_10/StatefulPartitionedCall.batch_normalization_10/StatefulPartitionedCall2`
.batch_normalization_11/StatefulPartitionedCall.batch_normalization_11/StatefulPartitionedCall2`
.batch_normalization_12/StatefulPartitionedCall.batch_normalization_12/StatefulPartitionedCall2`
.batch_normalization_13/StatefulPartitionedCall.batch_normalization_13/StatefulPartitionedCall2`
.batch_normalization_14/StatefulPartitionedCall.batch_normalization_14/StatefulPartitionedCall2F
!conv2d_10/StatefulPartitionedCall!conv2d_10/StatefulPartitionedCall2F
!conv2d_11/StatefulPartitionedCall!conv2d_11/StatefulPartitionedCall2F
!conv2d_12/StatefulPartitionedCall!conv2d_12/StatefulPartitionedCall2F
!conv2d_13/StatefulPartitionedCall!conv2d_13/StatefulPartitionedCall2F
!conv2d_14/StatefulPartitionedCall!conv2d_14/StatefulPartitionedCall2B
dense_6/StatefulPartitionedCalldense_6/StatefulPartitionedCall2B
dense_7/StatefulPartitionedCalldense_7/StatefulPartitionedCall2B
dense_8/StatefulPartitionedCalldense_8/StatefulPartitionedCall2F
!dropout_4/StatefulPartitionedCall!dropout_4/StatefulPartitionedCall2F
!dropout_5/StatefulPartitionedCall!dropout_5/StatefulPartitionedCall:d `
1
_output_shapes
:?????????เ
+
_user_specified_namerescaling_2_input
	
ี
6__inference_batch_normalization_11_layer_call_fn_93116

inputs
unknown:	
	unknown_0:	
	unknown_1:	
	unknown_2:	
identityขStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,???????????????????????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Z
fURS
Q__inference_batch_normalization_11_layer_call_and_return_conditional_losses_91253
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*B
_output_shapes0
.:,???????????????????????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,???????????????????????????: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,???????????????????????????
 
_user_specified_nameinputs
?
?
Q__inference_batch_normalization_12_layer_call_and_return_conditional_losses_91298

inputs&
readvariableop_resource:	(
readvariableop_1_resource:	7
(fusedbatchnormv3_readvariableop_resource:	9
*fusedbatchnormv3_readvariableop_1_resource:	
identityขFusedBatchNormV3/ReadVariableOpข!FusedBatchNormV3/ReadVariableOp_1ขReadVariableOpขReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:*
dtype0
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype0
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype0อ
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,???????????????????????????:::::*
epsilon%o:*
is_training( ~
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,???????????????????????????ฐ
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,???????????????????????????: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,???????????????????????????
 
_user_specified_nameinputs
?
?
Q__inference_batch_normalization_13_layer_call_and_return_conditional_losses_91362

inputs&
readvariableop_resource:	(
readvariableop_1_resource:	7
(fusedbatchnormv3_readvariableop_resource:	9
*fusedbatchnormv3_readvariableop_1_resource:	
identityขFusedBatchNormV3/ReadVariableOpข!FusedBatchNormV3/ReadVariableOp_1ขReadVariableOpขReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:*
dtype0
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype0
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype0อ
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,???????????????????????????:::::*
epsilon%o:*
is_training( ~
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,???????????????????????????ฐ
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,???????????????????????????: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,???????????????????????????
 
_user_specified_nameinputs
ส฿
๖"
G__inference_sequential_2_layer_call_and_return_conditional_losses_92965

inputsB
(conv2d_10_conv2d_readvariableop_resource:`7
)conv2d_10_biasadd_readvariableop_resource:`<
.batch_normalization_10_readvariableop_resource:`>
0batch_normalization_10_readvariableop_1_resource:`M
?batch_normalization_10_fusedbatchnormv3_readvariableop_resource:`O
Abatch_normalization_10_fusedbatchnormv3_readvariableop_1_resource:`C
(conv2d_11_conv2d_readvariableop_resource:`8
)conv2d_11_biasadd_readvariableop_resource:	=
.batch_normalization_11_readvariableop_resource:	?
0batch_normalization_11_readvariableop_1_resource:	N
?batch_normalization_11_fusedbatchnormv3_readvariableop_resource:	P
Abatch_normalization_11_fusedbatchnormv3_readvariableop_1_resource:	D
(conv2d_12_conv2d_readvariableop_resource:8
)conv2d_12_biasadd_readvariableop_resource:	=
.batch_normalization_12_readvariableop_resource:	?
0batch_normalization_12_readvariableop_1_resource:	N
?batch_normalization_12_fusedbatchnormv3_readvariableop_resource:	P
Abatch_normalization_12_fusedbatchnormv3_readvariableop_1_resource:	D
(conv2d_13_conv2d_readvariableop_resource:8
)conv2d_13_biasadd_readvariableop_resource:	=
.batch_normalization_13_readvariableop_resource:	?
0batch_normalization_13_readvariableop_1_resource:	N
?batch_normalization_13_fusedbatchnormv3_readvariableop_resource:	P
Abatch_normalization_13_fusedbatchnormv3_readvariableop_1_resource:	D
(conv2d_14_conv2d_readvariableop_resource:8
)conv2d_14_biasadd_readvariableop_resource:	=
.batch_normalization_14_readvariableop_resource:	?
0batch_normalization_14_readvariableop_1_resource:	N
?batch_normalization_14_fusedbatchnormv3_readvariableop_resource:	P
Abatch_normalization_14_fusedbatchnormv3_readvariableop_1_resource:	:
&dense_6_matmul_readvariableop_resource:
6
'dense_6_biasadd_readvariableop_resource:	:
&dense_7_matmul_readvariableop_resource:
6
'dense_7_biasadd_readvariableop_resource:	9
&dense_8_matmul_readvariableop_resource:	5
'dense_8_biasadd_readvariableop_resource:
identityข%batch_normalization_10/AssignNewValueข'batch_normalization_10/AssignNewValue_1ข6batch_normalization_10/FusedBatchNormV3/ReadVariableOpข8batch_normalization_10/FusedBatchNormV3/ReadVariableOp_1ข%batch_normalization_10/ReadVariableOpข'batch_normalization_10/ReadVariableOp_1ข%batch_normalization_11/AssignNewValueข'batch_normalization_11/AssignNewValue_1ข6batch_normalization_11/FusedBatchNormV3/ReadVariableOpข8batch_normalization_11/FusedBatchNormV3/ReadVariableOp_1ข%batch_normalization_11/ReadVariableOpข'batch_normalization_11/ReadVariableOp_1ข%batch_normalization_12/AssignNewValueข'batch_normalization_12/AssignNewValue_1ข6batch_normalization_12/FusedBatchNormV3/ReadVariableOpข8batch_normalization_12/FusedBatchNormV3/ReadVariableOp_1ข%batch_normalization_12/ReadVariableOpข'batch_normalization_12/ReadVariableOp_1ข%batch_normalization_13/AssignNewValueข'batch_normalization_13/AssignNewValue_1ข6batch_normalization_13/FusedBatchNormV3/ReadVariableOpข8batch_normalization_13/FusedBatchNormV3/ReadVariableOp_1ข%batch_normalization_13/ReadVariableOpข'batch_normalization_13/ReadVariableOp_1ข%batch_normalization_14/AssignNewValueข'batch_normalization_14/AssignNewValue_1ข6batch_normalization_14/FusedBatchNormV3/ReadVariableOpข8batch_normalization_14/FusedBatchNormV3/ReadVariableOp_1ข%batch_normalization_14/ReadVariableOpข'batch_normalization_14/ReadVariableOp_1ข conv2d_10/BiasAdd/ReadVariableOpขconv2d_10/Conv2D/ReadVariableOpข conv2d_11/BiasAdd/ReadVariableOpขconv2d_11/Conv2D/ReadVariableOpข conv2d_12/BiasAdd/ReadVariableOpขconv2d_12/Conv2D/ReadVariableOpข conv2d_13/BiasAdd/ReadVariableOpขconv2d_13/Conv2D/ReadVariableOpข conv2d_14/BiasAdd/ReadVariableOpขconv2d_14/Conv2D/ReadVariableOpขdense_6/BiasAdd/ReadVariableOpขdense_6/MatMul/ReadVariableOpขdense_7/BiasAdd/ReadVariableOpขdense_7/MatMul/ReadVariableOpขdense_8/BiasAdd/ReadVariableOpขdense_8/MatMul/ReadVariableOpW
rescaling_2/Cast/xConst*
_output_shapes
: *
dtype0*
valueB
 *;Y
rescaling_2/Cast_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *    w
rescaling_2/mulMulinputsrescaling_2/Cast/x:output:0*
T0*1
_output_shapes
:?????????เ
rescaling_2/addAddV2rescaling_2/mul:z:0rescaling_2/Cast_1/x:output:0*
T0*1
_output_shapes
:?????????เ
conv2d_10/Conv2D/ReadVariableOpReadVariableOp(conv2d_10_conv2d_readvariableop_resource*&
_output_shapes
:`*
dtype0ผ
conv2d_10/Conv2DConv2Drescaling_2/add:z:0'conv2d_10/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????v`*
paddingVALID*
strides

 conv2d_10/BiasAdd/ReadVariableOpReadVariableOp)conv2d_10_biasadd_readvariableop_resource*
_output_shapes
:`*
dtype0
conv2d_10/BiasAddBiasAddconv2d_10/Conv2D:output:0(conv2d_10/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????v`m
conv2d_10/ReluReluconv2d_10/BiasAdd:output:0*
T0*0
_output_shapes
:?????????v`
%batch_normalization_10/ReadVariableOpReadVariableOp.batch_normalization_10_readvariableop_resource*
_output_shapes
:`*
dtype0
'batch_normalization_10/ReadVariableOp_1ReadVariableOp0batch_normalization_10_readvariableop_1_resource*
_output_shapes
:`*
dtype0ฒ
6batch_normalization_10/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_10_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:`*
dtype0ถ
8batch_normalization_10/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_10_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:`*
dtype0ฮ
'batch_normalization_10/FusedBatchNormV3FusedBatchNormV3conv2d_10/Relu:activations:0-batch_normalization_10/ReadVariableOp:value:0/batch_normalization_10/ReadVariableOp_1:value:0>batch_normalization_10/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_10/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*L
_output_shapes:
8:?????????v`:`:`:`:`:*
epsilon%o:*
exponential_avg_factor%
ื#<ข
%batch_normalization_10/AssignNewValueAssignVariableOp?batch_normalization_10_fusedbatchnormv3_readvariableop_resource4batch_normalization_10/FusedBatchNormV3:batch_mean:07^batch_normalization_10/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(ฌ
'batch_normalization_10/AssignNewValue_1AssignVariableOpAbatch_normalization_10_fusedbatchnormv3_readvariableop_1_resource8batch_normalization_10/FusedBatchNormV3:batch_variance:09^batch_normalization_10/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(ผ
max_pooling2d_6/MaxPoolMaxPool+batch_normalization_10/FusedBatchNormV3:y:0*/
_output_shapes
:?????????:N`*
ksize
*
paddingVALID*
strides

conv2d_11/Conv2D/ReadVariableOpReadVariableOp(conv2d_11_conv2d_readvariableop_resource*'
_output_shapes
:`*
dtype0ศ
conv2d_11/Conv2DConv2D max_pooling2d_6/MaxPool:output:0'conv2d_11/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????*
paddingSAME*
strides

 conv2d_11/BiasAdd/ReadVariableOpReadVariableOp)conv2d_11_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0
conv2d_11/BiasAddBiasAddconv2d_11/Conv2D:output:0(conv2d_11/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????m
conv2d_11/ReluReluconv2d_11/BiasAdd:output:0*
T0*0
_output_shapes
:?????????
%batch_normalization_11/ReadVariableOpReadVariableOp.batch_normalization_11_readvariableop_resource*
_output_shapes	
:*
dtype0
'batch_normalization_11/ReadVariableOp_1ReadVariableOp0batch_normalization_11_readvariableop_1_resource*
_output_shapes	
:*
dtype0ณ
6batch_normalization_11/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_11_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype0ท
8batch_normalization_11/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_11_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype0า
'batch_normalization_11/FusedBatchNormV3FusedBatchNormV3conv2d_11/Relu:activations:0-batch_normalization_11/ReadVariableOp:value:0/batch_normalization_11/ReadVariableOp_1:value:0>batch_normalization_11/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_11/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:?????????:::::*
epsilon%o:*
exponential_avg_factor%
ื#<ข
%batch_normalization_11/AssignNewValueAssignVariableOp?batch_normalization_11_fusedbatchnormv3_readvariableop_resource4batch_normalization_11/FusedBatchNormV3:batch_mean:07^batch_normalization_11/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(ฌ
'batch_normalization_11/AssignNewValue_1AssignVariableOpAbatch_normalization_11_fusedbatchnormv3_readvariableop_1_resource8batch_normalization_11/FusedBatchNormV3:batch_variance:09^batch_normalization_11/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(ฝ
max_pooling2d_7/MaxPoolMaxPool+batch_normalization_11/FusedBatchNormV3:y:0*0
_output_shapes
:?????????	*
ksize
*
paddingVALID*
strides

conv2d_12/Conv2D/ReadVariableOpReadVariableOp(conv2d_12_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0ศ
conv2d_12/Conv2DConv2D max_pooling2d_7/MaxPool:output:0'conv2d_12/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????*
paddingSAME*
strides

 conv2d_12/BiasAdd/ReadVariableOpReadVariableOp)conv2d_12_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0
conv2d_12/BiasAddBiasAddconv2d_12/Conv2D:output:0(conv2d_12/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????m
conv2d_12/ReluReluconv2d_12/BiasAdd:output:0*
T0*0
_output_shapes
:?????????
%batch_normalization_12/ReadVariableOpReadVariableOp.batch_normalization_12_readvariableop_resource*
_output_shapes	
:*
dtype0
'batch_normalization_12/ReadVariableOp_1ReadVariableOp0batch_normalization_12_readvariableop_1_resource*
_output_shapes	
:*
dtype0ณ
6batch_normalization_12/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_12_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype0ท
8batch_normalization_12/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_12_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype0า
'batch_normalization_12/FusedBatchNormV3FusedBatchNormV3conv2d_12/Relu:activations:0-batch_normalization_12/ReadVariableOp:value:0/batch_normalization_12/ReadVariableOp_1:value:0>batch_normalization_12/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_12/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:?????????:::::*
epsilon%o:*
exponential_avg_factor%
ื#<ข
%batch_normalization_12/AssignNewValueAssignVariableOp?batch_normalization_12_fusedbatchnormv3_readvariableop_resource4batch_normalization_12/FusedBatchNormV3:batch_mean:07^batch_normalization_12/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(ฌ
'batch_normalization_12/AssignNewValue_1AssignVariableOpAbatch_normalization_12_fusedbatchnormv3_readvariableop_1_resource8batch_normalization_12/FusedBatchNormV3:batch_variance:09^batch_normalization_12/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(
conv2d_13/Conv2D/ReadVariableOpReadVariableOp(conv2d_13_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0ำ
conv2d_13/Conv2DConv2D+batch_normalization_12/FusedBatchNormV3:y:0'conv2d_13/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????*
paddingSAME*
strides

 conv2d_13/BiasAdd/ReadVariableOpReadVariableOp)conv2d_13_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0
conv2d_13/BiasAddBiasAddconv2d_13/Conv2D:output:0(conv2d_13/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????m
conv2d_13/ReluReluconv2d_13/BiasAdd:output:0*
T0*0
_output_shapes
:?????????
%batch_normalization_13/ReadVariableOpReadVariableOp.batch_normalization_13_readvariableop_resource*
_output_shapes	
:*
dtype0
'batch_normalization_13/ReadVariableOp_1ReadVariableOp0batch_normalization_13_readvariableop_1_resource*
_output_shapes	
:*
dtype0ณ
6batch_normalization_13/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_13_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype0ท
8batch_normalization_13/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_13_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype0า
'batch_normalization_13/FusedBatchNormV3FusedBatchNormV3conv2d_13/Relu:activations:0-batch_normalization_13/ReadVariableOp:value:0/batch_normalization_13/ReadVariableOp_1:value:0>batch_normalization_13/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_13/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:?????????:::::*
epsilon%o:*
exponential_avg_factor%
ื#<ข
%batch_normalization_13/AssignNewValueAssignVariableOp?batch_normalization_13_fusedbatchnormv3_readvariableop_resource4batch_normalization_13/FusedBatchNormV3:batch_mean:07^batch_normalization_13/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(ฌ
'batch_normalization_13/AssignNewValue_1AssignVariableOpAbatch_normalization_13_fusedbatchnormv3_readvariableop_1_resource8batch_normalization_13/FusedBatchNormV3:batch_variance:09^batch_normalization_13/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(
conv2d_14/Conv2D/ReadVariableOpReadVariableOp(conv2d_14_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0ำ
conv2d_14/Conv2DConv2D+batch_normalization_13/FusedBatchNormV3:y:0'conv2d_14/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????*
paddingSAME*
strides

 conv2d_14/BiasAdd/ReadVariableOpReadVariableOp)conv2d_14_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0
conv2d_14/BiasAddBiasAddconv2d_14/Conv2D:output:0(conv2d_14/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????m
conv2d_14/ReluReluconv2d_14/BiasAdd:output:0*
T0*0
_output_shapes
:?????????
%batch_normalization_14/ReadVariableOpReadVariableOp.batch_normalization_14_readvariableop_resource*
_output_shapes	
:*
dtype0
'batch_normalization_14/ReadVariableOp_1ReadVariableOp0batch_normalization_14_readvariableop_1_resource*
_output_shapes	
:*
dtype0ณ
6batch_normalization_14/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_14_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype0ท
8batch_normalization_14/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_14_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype0า
'batch_normalization_14/FusedBatchNormV3FusedBatchNormV3conv2d_14/Relu:activations:0-batch_normalization_14/ReadVariableOp:value:0/batch_normalization_14/ReadVariableOp_1:value:0>batch_normalization_14/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_14/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:?????????:::::*
epsilon%o:*
exponential_avg_factor%
ื#<ข
%batch_normalization_14/AssignNewValueAssignVariableOp?batch_normalization_14_fusedbatchnormv3_readvariableop_resource4batch_normalization_14/FusedBatchNormV3:batch_mean:07^batch_normalization_14/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(ฌ
'batch_normalization_14/AssignNewValue_1AssignVariableOpAbatch_normalization_14_fusedbatchnormv3_readvariableop_1_resource8batch_normalization_14/FusedBatchNormV3:batch_variance:09^batch_normalization_14/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(ฝ
max_pooling2d_8/MaxPoolMaxPool+batch_normalization_14/FusedBatchNormV3:y:0*0
_output_shapes
:?????????*
ksize
*
paddingVALID*
strides
`
flatten_2/ConstConst*
_output_shapes
:*
dtype0*
valueB"????   
flatten_2/ReshapeReshape max_pooling2d_8/MaxPool:output:0flatten_2/Const:output:0*
T0*(
_output_shapes
:?????????
dense_6/MatMul/ReadVariableOpReadVariableOp&dense_6_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype0
dense_6/MatMulMatMulflatten_2/Reshape:output:0%dense_6/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????
dense_6/BiasAdd/ReadVariableOpReadVariableOp'dense_6_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0
dense_6/BiasAddBiasAdddense_6/MatMul:product:0&dense_6/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????a
dense_6/ReluReludense_6/BiasAdd:output:0*
T0*(
_output_shapes
:?????????\
dropout_4/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @
dropout_4/dropout/MulMuldense_6/Relu:activations:0 dropout_4/dropout/Const:output:0*
T0*(
_output_shapes
:?????????a
dropout_4/dropout/ShapeShapedense_6/Relu:activations:0*
T0*
_output_shapes
:ก
.dropout_4/dropout/random_uniform/RandomUniformRandomUniform dropout_4/dropout/Shape:output:0*
T0*(
_output_shapes
:?????????*
dtype0e
 dropout_4/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?ล
dropout_4/dropout/GreaterEqualGreaterEqual7dropout_4/dropout/random_uniform/RandomUniform:output:0)dropout_4/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:?????????
dropout_4/dropout/CastCast"dropout_4/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:?????????
dropout_4/dropout/Mul_1Muldropout_4/dropout/Mul:z:0dropout_4/dropout/Cast:y:0*
T0*(
_output_shapes
:?????????
dense_7/MatMul/ReadVariableOpReadVariableOp&dense_7_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype0
dense_7/MatMulMatMuldropout_4/dropout/Mul_1:z:0%dense_7/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????
dense_7/BiasAdd/ReadVariableOpReadVariableOp'dense_7_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0
dense_7/BiasAddBiasAdddense_7/MatMul:product:0&dense_7/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????a
dense_7/ReluReludense_7/BiasAdd:output:0*
T0*(
_output_shapes
:?????????\
dropout_5/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @
dropout_5/dropout/MulMuldense_7/Relu:activations:0 dropout_5/dropout/Const:output:0*
T0*(
_output_shapes
:?????????a
dropout_5/dropout/ShapeShapedense_7/Relu:activations:0*
T0*
_output_shapes
:ก
.dropout_5/dropout/random_uniform/RandomUniformRandomUniform dropout_5/dropout/Shape:output:0*
T0*(
_output_shapes
:?????????*
dtype0e
 dropout_5/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?ล
dropout_5/dropout/GreaterEqualGreaterEqual7dropout_5/dropout/random_uniform/RandomUniform:output:0)dropout_5/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:?????????
dropout_5/dropout/CastCast"dropout_5/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:?????????
dropout_5/dropout/Mul_1Muldropout_5/dropout/Mul:z:0dropout_5/dropout/Cast:y:0*
T0*(
_output_shapes
:?????????
dense_8/MatMul/ReadVariableOpReadVariableOp&dense_8_matmul_readvariableop_resource*
_output_shapes
:	*
dtype0
dense_8/MatMulMatMuldropout_5/dropout/Mul_1:z:0%dense_8/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
dense_8/BiasAdd/ReadVariableOpReadVariableOp'dense_8_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense_8/BiasAddBiasAdddense_8/MatMul:product:0&dense_8/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????f
dense_8/SoftmaxSoftmaxdense_8/BiasAdd:output:0*
T0*'
_output_shapes
:?????????h
IdentityIdentitydense_8/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:?????????ฺ
NoOpNoOp&^batch_normalization_10/AssignNewValue(^batch_normalization_10/AssignNewValue_17^batch_normalization_10/FusedBatchNormV3/ReadVariableOp9^batch_normalization_10/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_10/ReadVariableOp(^batch_normalization_10/ReadVariableOp_1&^batch_normalization_11/AssignNewValue(^batch_normalization_11/AssignNewValue_17^batch_normalization_11/FusedBatchNormV3/ReadVariableOp9^batch_normalization_11/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_11/ReadVariableOp(^batch_normalization_11/ReadVariableOp_1&^batch_normalization_12/AssignNewValue(^batch_normalization_12/AssignNewValue_17^batch_normalization_12/FusedBatchNormV3/ReadVariableOp9^batch_normalization_12/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_12/ReadVariableOp(^batch_normalization_12/ReadVariableOp_1&^batch_normalization_13/AssignNewValue(^batch_normalization_13/AssignNewValue_17^batch_normalization_13/FusedBatchNormV3/ReadVariableOp9^batch_normalization_13/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_13/ReadVariableOp(^batch_normalization_13/ReadVariableOp_1&^batch_normalization_14/AssignNewValue(^batch_normalization_14/AssignNewValue_17^batch_normalization_14/FusedBatchNormV3/ReadVariableOp9^batch_normalization_14/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_14/ReadVariableOp(^batch_normalization_14/ReadVariableOp_1!^conv2d_10/BiasAdd/ReadVariableOp ^conv2d_10/Conv2D/ReadVariableOp!^conv2d_11/BiasAdd/ReadVariableOp ^conv2d_11/Conv2D/ReadVariableOp!^conv2d_12/BiasAdd/ReadVariableOp ^conv2d_12/Conv2D/ReadVariableOp!^conv2d_13/BiasAdd/ReadVariableOp ^conv2d_13/Conv2D/ReadVariableOp!^conv2d_14/BiasAdd/ReadVariableOp ^conv2d_14/Conv2D/ReadVariableOp^dense_6/BiasAdd/ReadVariableOp^dense_6/MatMul/ReadVariableOp^dense_7/BiasAdd/ReadVariableOp^dense_7/MatMul/ReadVariableOp^dense_8/BiasAdd/ReadVariableOp^dense_8/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*x
_input_shapesg
e:?????????เ: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2N
%batch_normalization_10/AssignNewValue%batch_normalization_10/AssignNewValue2R
'batch_normalization_10/AssignNewValue_1'batch_normalization_10/AssignNewValue_12p
6batch_normalization_10/FusedBatchNormV3/ReadVariableOp6batch_normalization_10/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_10/FusedBatchNormV3/ReadVariableOp_18batch_normalization_10/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_10/ReadVariableOp%batch_normalization_10/ReadVariableOp2R
'batch_normalization_10/ReadVariableOp_1'batch_normalization_10/ReadVariableOp_12N
%batch_normalization_11/AssignNewValue%batch_normalization_11/AssignNewValue2R
'batch_normalization_11/AssignNewValue_1'batch_normalization_11/AssignNewValue_12p
6batch_normalization_11/FusedBatchNormV3/ReadVariableOp6batch_normalization_11/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_11/FusedBatchNormV3/ReadVariableOp_18batch_normalization_11/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_11/ReadVariableOp%batch_normalization_11/ReadVariableOp2R
'batch_normalization_11/ReadVariableOp_1'batch_normalization_11/ReadVariableOp_12N
%batch_normalization_12/AssignNewValue%batch_normalization_12/AssignNewValue2R
'batch_normalization_12/AssignNewValue_1'batch_normalization_12/AssignNewValue_12p
6batch_normalization_12/FusedBatchNormV3/ReadVariableOp6batch_normalization_12/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_12/FusedBatchNormV3/ReadVariableOp_18batch_normalization_12/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_12/ReadVariableOp%batch_normalization_12/ReadVariableOp2R
'batch_normalization_12/ReadVariableOp_1'batch_normalization_12/ReadVariableOp_12N
%batch_normalization_13/AssignNewValue%batch_normalization_13/AssignNewValue2R
'batch_normalization_13/AssignNewValue_1'batch_normalization_13/AssignNewValue_12p
6batch_normalization_13/FusedBatchNormV3/ReadVariableOp6batch_normalization_13/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_13/FusedBatchNormV3/ReadVariableOp_18batch_normalization_13/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_13/ReadVariableOp%batch_normalization_13/ReadVariableOp2R
'batch_normalization_13/ReadVariableOp_1'batch_normalization_13/ReadVariableOp_12N
%batch_normalization_14/AssignNewValue%batch_normalization_14/AssignNewValue2R
'batch_normalization_14/AssignNewValue_1'batch_normalization_14/AssignNewValue_12p
6batch_normalization_14/FusedBatchNormV3/ReadVariableOp6batch_normalization_14/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_14/FusedBatchNormV3/ReadVariableOp_18batch_normalization_14/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_14/ReadVariableOp%batch_normalization_14/ReadVariableOp2R
'batch_normalization_14/ReadVariableOp_1'batch_normalization_14/ReadVariableOp_12D
 conv2d_10/BiasAdd/ReadVariableOp conv2d_10/BiasAdd/ReadVariableOp2B
conv2d_10/Conv2D/ReadVariableOpconv2d_10/Conv2D/ReadVariableOp2D
 conv2d_11/BiasAdd/ReadVariableOp conv2d_11/BiasAdd/ReadVariableOp2B
conv2d_11/Conv2D/ReadVariableOpconv2d_11/Conv2D/ReadVariableOp2D
 conv2d_12/BiasAdd/ReadVariableOp conv2d_12/BiasAdd/ReadVariableOp2B
conv2d_12/Conv2D/ReadVariableOpconv2d_12/Conv2D/ReadVariableOp2D
 conv2d_13/BiasAdd/ReadVariableOp conv2d_13/BiasAdd/ReadVariableOp2B
conv2d_13/Conv2D/ReadVariableOpconv2d_13/Conv2D/ReadVariableOp2D
 conv2d_14/BiasAdd/ReadVariableOp conv2d_14/BiasAdd/ReadVariableOp2B
conv2d_14/Conv2D/ReadVariableOpconv2d_14/Conv2D/ReadVariableOp2@
dense_6/BiasAdd/ReadVariableOpdense_6/BiasAdd/ReadVariableOp2>
dense_6/MatMul/ReadVariableOpdense_6/MatMul/ReadVariableOp2@
dense_7/BiasAdd/ReadVariableOpdense_7/BiasAdd/ReadVariableOp2>
dense_7/MatMul/ReadVariableOpdense_7/MatMul/ReadVariableOp2@
dense_8/BiasAdd/ReadVariableOpdense_8/BiasAdd/ReadVariableOp2>
dense_8/MatMul/ReadVariableOpdense_8/MatMul/ReadVariableOp:Y U
1
_output_shapes
:?????????เ
 
_user_specified_nameinputs
๐

)__inference_conv2d_10_layer_call_fn_92987

inputs!
unknown:`
	unknown_0:`
identityขStatefulPartitionedCallโ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????v`*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_conv2d_10_layer_call_and_return_conditional_losses_91508x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:?????????v``
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:?????????เ: : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:?????????เ
 
_user_specified_nameinputs

?
D__inference_conv2d_10_layer_call_and_return_conditional_losses_91508

inputs8
conv2d_readvariableop_resource:`-
biasadd_readvariableop_resource:`
identityขBiasAdd/ReadVariableOpขConv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:`*
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????v`*
paddingVALID*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:`*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????v`Y
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:?????????v`j
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:?????????v`w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:?????????เ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:?????????เ
 
_user_specified_nameinputs

?
D__inference_conv2d_11_layer_call_and_return_conditional_losses_91535

inputs9
conv2d_readvariableop_resource:`.
biasadd_readvariableop_resource:	
identityขBiasAdd/ReadVariableOpขConv2D/ReadVariableOp}
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:`*
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????*
paddingSAME*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????Y
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:?????????j
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:?????????w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????:N`: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????:N`
 
_user_specified_nameinputs
บ
๊
,__inference_sequential_2_layer_call_fn_92669

inputs!
unknown:`
	unknown_0:`
	unknown_1:`
	unknown_2:`
	unknown_3:`
	unknown_4:`$
	unknown_5:`
	unknown_6:	
	unknown_7:	
	unknown_8:	
	unknown_9:	

unknown_10:	&

unknown_11:

unknown_12:	

unknown_13:	

unknown_14:	

unknown_15:	

unknown_16:	&

unknown_17:

unknown_18:	

unknown_19:	

unknown_20:	

unknown_21:	

unknown_22:	&

unknown_23:

unknown_24:	

unknown_25:	

unknown_26:	

unknown_27:	

unknown_28:	

unknown_29:


unknown_30:	

unknown_31:


unknown_32:	

unknown_33:	

unknown_34:
identityขStatefulPartitionedCallฅ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34*0
Tin)
'2%*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*<
_read_only_resource_inputs
	
 !"#$*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_sequential_2_layer_call_and_return_conditional_losses_92092o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*x
_input_shapesg
e:?????????เ: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:?????????เ
 
_user_specified_nameinputs


D__inference_conv2d_14_layer_call_and_return_conditional_losses_91614

inputs:
conv2d_readvariableop_resource:.
biasadd_readvariableop_resource:	
identityขBiasAdd/ReadVariableOpขConv2D/ReadVariableOp~
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:*
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????*
paddingSAME*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????Y
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:?????????j
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:?????????w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :?????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:?????????
 
_user_specified_nameinputs
ษ
G
+__inference_rescaling_2_layer_call_fn_92970

inputs
identityป
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:?????????เ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_rescaling_2_layer_call_and_return_conditional_losses_91495j
IdentityIdentityPartitionedCall:output:0*
T0*1
_output_shapes
:?????????เ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:?????????เ:Y U
1
_output_shapes
:?????????เ
 
_user_specified_nameinputs

ฤ
Q__inference_batch_normalization_13_layer_call_and_return_conditional_losses_93326

inputs&
readvariableop_resource:	(
readvariableop_1_resource:	7
(fusedbatchnormv3_readvariableop_resource:	9
*fusedbatchnormv3_readvariableop_1_resource:	
identityขAssignNewValueขAssignNewValue_1ขFusedBatchNormV3/ReadVariableOpข!FusedBatchNormV3/ReadVariableOp_1ขReadVariableOpขReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:*
dtype0
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype0
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype0?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,???????????????????????????:::::*
epsilon%o:*
exponential_avg_factor%
ื#<ฦ
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(ะ
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(~
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,???????????????????????????ิ
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,???????????????????????????: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,???????????????????????????
 
_user_specified_nameinputs
ถ
K
/__inference_max_pooling2d_6_layer_call_fn_93065

inputs
identityุ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4????????????????????????????????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *S
fNRL
J__inference_max_pooling2d_6_layer_call_and_return_conditional_losses_91197
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs

?
D__inference_conv2d_10_layer_call_and_return_conditional_losses_92998

inputs8
conv2d_readvariableop_resource:`-
biasadd_readvariableop_resource:`
identityขBiasAdd/ReadVariableOpขConv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:`*
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????v`*
paddingVALID*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:`*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????v`Y
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:?????????v`j
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:?????????v`w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:?????????เ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:?????????เ
 
_user_specified_nameinputs

?
D__inference_conv2d_11_layer_call_and_return_conditional_losses_93090

inputs9
conv2d_readvariableop_resource:`.
biasadd_readvariableop_resource:	
identityขBiasAdd/ReadVariableOpขConv2D/ReadVariableOp}
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:`*
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????*
paddingSAME*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????Y
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:?????????j
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:?????????w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????:N`: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????:N`
 
_user_specified_nameinputs
S

__inference__traced_save_93686
file_prefix/
+savev2_conv2d_10_kernel_read_readvariableop-
)savev2_conv2d_10_bias_read_readvariableop;
7savev2_batch_normalization_10_gamma_read_readvariableop:
6savev2_batch_normalization_10_beta_read_readvariableopA
=savev2_batch_normalization_10_moving_mean_read_readvariableopE
Asavev2_batch_normalization_10_moving_variance_read_readvariableop/
+savev2_conv2d_11_kernel_read_readvariableop-
)savev2_conv2d_11_bias_read_readvariableop;
7savev2_batch_normalization_11_gamma_read_readvariableop:
6savev2_batch_normalization_11_beta_read_readvariableopA
=savev2_batch_normalization_11_moving_mean_read_readvariableopE
Asavev2_batch_normalization_11_moving_variance_read_readvariableop/
+savev2_conv2d_12_kernel_read_readvariableop-
)savev2_conv2d_12_bias_read_readvariableop;
7savev2_batch_normalization_12_gamma_read_readvariableop:
6savev2_batch_normalization_12_beta_read_readvariableopA
=savev2_batch_normalization_12_moving_mean_read_readvariableopE
Asavev2_batch_normalization_12_moving_variance_read_readvariableop/
+savev2_conv2d_13_kernel_read_readvariableop-
)savev2_conv2d_13_bias_read_readvariableop;
7savev2_batch_normalization_13_gamma_read_readvariableop:
6savev2_batch_normalization_13_beta_read_readvariableopA
=savev2_batch_normalization_13_moving_mean_read_readvariableopE
Asavev2_batch_normalization_13_moving_variance_read_readvariableop/
+savev2_conv2d_14_kernel_read_readvariableop-
)savev2_conv2d_14_bias_read_readvariableop;
7savev2_batch_normalization_14_gamma_read_readvariableop:
6savev2_batch_normalization_14_beta_read_readvariableopA
=savev2_batch_normalization_14_moving_mean_read_readvariableopE
Asavev2_batch_normalization_14_moving_variance_read_readvariableop-
)savev2_dense_6_kernel_read_readvariableop+
'savev2_dense_6_bias_read_readvariableop-
)savev2_dense_7_kernel_read_readvariableop+
'savev2_dense_7_bias_read_readvariableop-
)savev2_dense_8_kernel_read_readvariableop+
'savev2_dense_8_bias_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop
savev2_const

identity_1ขMergeV2Checkpointsw
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*Z
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.parta
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: f

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: L

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: ๏
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:)*
dtype0*
valueB)B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-1/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-1/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-3/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-3/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-5/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-5/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-5/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-7/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-7/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-7/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-9/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-9/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-9/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-9/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-10/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-10/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-11/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-11/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-12/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-12/bias/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPHฟ
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:)*
dtype0*e
value\BZ)B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B ื
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0+savev2_conv2d_10_kernel_read_readvariableop)savev2_conv2d_10_bias_read_readvariableop7savev2_batch_normalization_10_gamma_read_readvariableop6savev2_batch_normalization_10_beta_read_readvariableop=savev2_batch_normalization_10_moving_mean_read_readvariableopAsavev2_batch_normalization_10_moving_variance_read_readvariableop+savev2_conv2d_11_kernel_read_readvariableop)savev2_conv2d_11_bias_read_readvariableop7savev2_batch_normalization_11_gamma_read_readvariableop6savev2_batch_normalization_11_beta_read_readvariableop=savev2_batch_normalization_11_moving_mean_read_readvariableopAsavev2_batch_normalization_11_moving_variance_read_readvariableop+savev2_conv2d_12_kernel_read_readvariableop)savev2_conv2d_12_bias_read_readvariableop7savev2_batch_normalization_12_gamma_read_readvariableop6savev2_batch_normalization_12_beta_read_readvariableop=savev2_batch_normalization_12_moving_mean_read_readvariableopAsavev2_batch_normalization_12_moving_variance_read_readvariableop+savev2_conv2d_13_kernel_read_readvariableop)savev2_conv2d_13_bias_read_readvariableop7savev2_batch_normalization_13_gamma_read_readvariableop6savev2_batch_normalization_13_beta_read_readvariableop=savev2_batch_normalization_13_moving_mean_read_readvariableopAsavev2_batch_normalization_13_moving_variance_read_readvariableop+savev2_conv2d_14_kernel_read_readvariableop)savev2_conv2d_14_bias_read_readvariableop7savev2_batch_normalization_14_gamma_read_readvariableop6savev2_batch_normalization_14_beta_read_readvariableop=savev2_batch_normalization_14_moving_mean_read_readvariableopAsavev2_batch_normalization_14_moving_variance_read_readvariableop)savev2_dense_6_kernel_read_readvariableop'savev2_dense_6_bias_read_readvariableop)savev2_dense_7_kernel_read_readvariableop'savev2_dense_7_bias_read_readvariableop)savev2_dense_8_kernel_read_readvariableop'savev2_dense_8_bias_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *7
dtypes-
+2)
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 f
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: Q

Identity_1IdentityIdentity:output:0^NoOp*
T0*
_output_shapes
: [
NoOpNoOp^MergeV2Checkpoints*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0*ใ
_input_shapesั
ฮ: :`:`:`:`:`:`:`::::::::::::::::::::::::
::
::	:: : : : : 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:,(
&
_output_shapes
:`: 

_output_shapes
:`: 

_output_shapes
:`: 

_output_shapes
:`: 

_output_shapes
:`: 

_output_shapes
:`:-)
'
_output_shapes
:`:!

_output_shapes	
::!	

_output_shapes	
::!


_output_shapes	
::!

_output_shapes	
::!

_output_shapes	
::.*
(
_output_shapes
::!

_output_shapes	
::!

_output_shapes	
::!

_output_shapes	
::!

_output_shapes	
::!

_output_shapes	
::.*
(
_output_shapes
::!

_output_shapes	
::!

_output_shapes	
::!

_output_shapes	
::!

_output_shapes	
::!

_output_shapes	
::.*
(
_output_shapes
::!

_output_shapes	
::!

_output_shapes	
::!

_output_shapes	
::!

_output_shapes	
::!

_output_shapes	
::&"
 
_output_shapes
:
:! 

_output_shapes	
::&!"
 
_output_shapes
:
:!"

_output_shapes	
::%#!

_output_shapes
:	: $

_output_shapes
::%

_output_shapes
: :&

_output_shapes
: :'

_output_shapes
: :(

_output_shapes
: :)

_output_shapes
: 
ถ
K
/__inference_max_pooling2d_8_layer_call_fn_93413

inputs
identityุ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4????????????????????????????????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *S
fNRL
J__inference_max_pooling2d_8_layer_call_and_return_conditional_losses_91477
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs

f
J__inference_max_pooling2d_6_layer_call_and_return_conditional_losses_93070

inputs
identityข
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4????????????????????????????????????*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs

f
J__inference_max_pooling2d_8_layer_call_and_return_conditional_losses_93418

inputs
identityข
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4????????????????????????????????????*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
ล

'__inference_dense_7_layer_call_fn_93485

inputs
unknown:

	unknown_0:	
identityขStatefulPartitionedCallุ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_dense_7_layer_call_and_return_conditional_losses_91673p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:?????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:?????????
 
_user_specified_nameinputs

ฤ
Q__inference_batch_normalization_14_layer_call_and_return_conditional_losses_93408

inputs&
readvariableop_resource:	(
readvariableop_1_resource:	7
(fusedbatchnormv3_readvariableop_resource:	9
*fusedbatchnormv3_readvariableop_1_resource:	
identityขAssignNewValueขAssignNewValue_1ขFusedBatchNormV3/ReadVariableOpข!FusedBatchNormV3/ReadVariableOp_1ขReadVariableOpขReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:*
dtype0
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype0
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype0?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,???????????????????????????:::::*
epsilon%o:*
exponential_avg_factor%
ื#<ฦ
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(ะ
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(~
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,???????????????????????????ิ
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,???????????????????????????: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,???????????????????????????
 
_user_specified_nameinputs

ฤ
Q__inference_batch_normalization_11_layer_call_and_return_conditional_losses_93152

inputs&
readvariableop_resource:	(
readvariableop_1_resource:	7
(fusedbatchnormv3_readvariableop_resource:	9
*fusedbatchnormv3_readvariableop_1_resource:	
identityขAssignNewValueขAssignNewValue_1ขFusedBatchNormV3/ReadVariableOpข!FusedBatchNormV3/ReadVariableOp_1ขReadVariableOpขReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:*
dtype0
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype0
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype0?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,???????????????????????????:::::*
epsilon%o:*
exponential_avg_factor%
ื#<ฦ
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(ะ
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(~
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,???????????????????????????ิ
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,???????????????????????????: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,???????????????????????????
 
_user_specified_nameinputs
ก
E
)__inference_dropout_5_layer_call_fn_93501

inputs
identityฐ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dropout_5_layer_call_and_return_conditional_losses_91684a
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:?????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:?????????:P L
(
_output_shapes
:?????????
 
_user_specified_nameinputs
๘
b
F__inference_rescaling_2_layer_call_and_return_conditional_losses_92978

inputs
identityK
Cast/xConst*
_output_shapes
: *
dtype0*
valueB
 *;M
Cast_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *    _
mulMulinputsCast/x:output:0*
T0*1
_output_shapes
:?????????เd
addAddV2mul:z:0Cast_1/x:output:0*
T0*1
_output_shapes
:?????????เY
IdentityIdentityadd:z:0*
T0*1
_output_shapes
:?????????เ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:?????????เ:Y U
1
_output_shapes
:?????????เ
 
_user_specified_nameinputs
?
b
D__inference_dropout_4_layer_call_and_return_conditional_losses_93464

inputs

identity_1O
IdentityIdentityinputs*
T0*(
_output_shapes
:?????????\

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:?????????"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:?????????:P L
(
_output_shapes
:?????????
 
_user_specified_nameinputs
	
ี
6__inference_batch_normalization_12_layer_call_fn_93195

inputs
unknown:	
	unknown_0:	
	unknown_1:	
	unknown_2:	
identityขStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,???????????????????????????*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Z
fURS
Q__inference_batch_normalization_12_layer_call_and_return_conditional_losses_91298
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*B
_output_shapes0
.:,???????????????????????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,???????????????????????????: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,???????????????????????????
 
_user_specified_nameinputs
?
๕
,__inference_sequential_2_layer_call_fn_92244
rescaling_2_input!
unknown:`
	unknown_0:`
	unknown_1:`
	unknown_2:`
	unknown_3:`
	unknown_4:`$
	unknown_5:`
	unknown_6:	
	unknown_7:	
	unknown_8:	
	unknown_9:	

unknown_10:	&

unknown_11:

unknown_12:	

unknown_13:	

unknown_14:	

unknown_15:	

unknown_16:	&

unknown_17:

unknown_18:	

unknown_19:	

unknown_20:	

unknown_21:	

unknown_22:	&

unknown_23:

unknown_24:	

unknown_25:	

unknown_26:	

unknown_27:	

unknown_28:	

unknown_29:


unknown_30:	

unknown_31:


unknown_32:	

unknown_33:	

unknown_34:
identityขStatefulPartitionedCallฐ
StatefulPartitionedCallStatefulPartitionedCallrescaling_2_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34*0
Tin)
'2%*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*<
_read_only_resource_inputs
	
 !"#$*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_sequential_2_layer_call_and_return_conditional_losses_92092o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*x
_input_shapesg
e:?????????เ: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:d `
1
_output_shapes
:?????????เ
+
_user_specified_namerescaling_2_input
๓
b
)__inference_dropout_5_layer_call_fn_93506

inputs
identityขStatefulPartitionedCallภ
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dropout_5_layer_call_and_return_conditional_losses_91809p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:?????????22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:?????????
 
_user_specified_nameinputs
ก
E
)__inference_dropout_4_layer_call_fn_93454

inputs
identityฐ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dropout_4_layer_call_and_return_conditional_losses_91660a
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:?????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:?????????:P L
(
_output_shapes
:?????????
 
_user_specified_nameinputs
ๅ
๕
,__inference_sequential_2_layer_call_fn_91779
rescaling_2_input!
unknown:`
	unknown_0:`
	unknown_1:`
	unknown_2:`
	unknown_3:`
	unknown_4:`$
	unknown_5:`
	unknown_6:	
	unknown_7:	
	unknown_8:	
	unknown_9:	

unknown_10:	&

unknown_11:

unknown_12:	

unknown_13:	

unknown_14:	

unknown_15:	

unknown_16:	&

unknown_17:

unknown_18:	

unknown_19:	

unknown_20:	

unknown_21:	

unknown_22:	&

unknown_23:

unknown_24:	

unknown_25:	

unknown_26:	

unknown_27:	

unknown_28:	

unknown_29:


unknown_30:	

unknown_31:


unknown_32:	

unknown_33:	

unknown_34:
identityขStatefulPartitionedCallบ
StatefulPartitionedCallStatefulPartitionedCallrescaling_2_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34*0
Tin)
'2%*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*F
_read_only_resource_inputs(
&$	
 !"#$*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_sequential_2_layer_call_and_return_conditional_losses_91704o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*x
_input_shapesg
e:?????????เ: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:d `
1
_output_shapes
:?????????เ
+
_user_specified_namerescaling_2_input
๑
ก
)__inference_conv2d_13_layer_call_fn_93253

inputs#
unknown:
	unknown_0:	
identityขStatefulPartitionedCallโ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_conv2d_13_layer_call_and_return_conditional_losses_91588x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :?????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:?????????
 
_user_specified_nameinputs

ภ
Q__inference_batch_normalization_10_layer_call_and_return_conditional_losses_93060

inputs%
readvariableop_resource:`'
readvariableop_1_resource:`6
(fusedbatchnormv3_readvariableop_resource:`8
*fusedbatchnormv3_readvariableop_1_resource:`
identityขAssignNewValueขAssignNewValue_1ขFusedBatchNormV3/ReadVariableOpข!FusedBatchNormV3/ReadVariableOp_1ขReadVariableOpขReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:`*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:`*
dtype0
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:`*
dtype0
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:`*
dtype0ึ
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????`:`:`:`:`:*
epsilon%o:*
exponential_avg_factor%
ื#<ฦ
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(ะ
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(}
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????`ิ
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????`: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????`
 
_user_specified_nameinputs
?
b
D__inference_dropout_5_layer_call_and_return_conditional_losses_93511

inputs

identity_1O
IdentityIdentityinputs*
T0*(
_output_shapes
:?????????\

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:?????????"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:?????????:P L
(
_output_shapes
:?????????
 
_user_specified_nameinputs
๚	
c
D__inference_dropout_4_layer_call_and_return_conditional_losses_93476

inputs
identityR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @e
dropout/MulMulinputsdropout/Const:output:0*
T0*(
_output_shapes
:?????????C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:?????????*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?ง
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:?????????p
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:?????????j
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*(
_output_shapes
:?????????Z
IdentityIdentitydropout/Mul_1:z:0*
T0*(
_output_shapes
:?????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:?????????:P L
(
_output_shapes
:?????????
 
_user_specified_nameinputs
	
ั
6__inference_batch_normalization_10_layer_call_fn_93024

inputs
unknown:`
	unknown_0:`
	unknown_1:`
	unknown_2:`
identityขStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????`*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Z
fURS
Q__inference_batch_normalization_10_layer_call_and_return_conditional_losses_91177
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????``
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????`: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+???????????????????????????`
 
_user_specified_nameinputs
	
ี
6__inference_batch_normalization_12_layer_call_fn_93208

inputs
unknown:	
	unknown_0:	
	unknown_1:	
	unknown_2:	
identityขStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,???????????????????????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Z
fURS
Q__inference_batch_normalization_12_layer_call_and_return_conditional_losses_91329
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*B
_output_shapes0
.:,???????????????????????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,???????????????????????????: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,???????????????????????????
 
_user_specified_nameinputs

ฤ
Q__inference_batch_normalization_12_layer_call_and_return_conditional_losses_91329

inputs&
readvariableop_resource:	(
readvariableop_1_resource:	7
(fusedbatchnormv3_readvariableop_resource:	9
*fusedbatchnormv3_readvariableop_1_resource:	
identityขAssignNewValueขAssignNewValue_1ขFusedBatchNormV3/ReadVariableOpข!FusedBatchNormV3/ReadVariableOp_1ขReadVariableOpขReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:*
dtype0
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype0
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype0?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,???????????????????????????:::::*
epsilon%o:*
exponential_avg_factor%
ื#<ฦ
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(ะ
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(~
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,???????????????????????????ิ
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,???????????????????????????: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,???????????????????????????
 
_user_specified_nameinputs
ฅ

๖
B__inference_dense_6_layer_call_and_return_conditional_losses_93449

inputs2
matmul_readvariableop_resource:
.
biasadd_readvariableop_resource:	
identityขBiasAdd/ReadVariableOpขMatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????Q
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:?????????b
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:?????????w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:?????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:?????????
 
_user_specified_nameinputs
?
b
D__inference_dropout_5_layer_call_and_return_conditional_losses_91684

inputs

identity_1O
IdentityIdentityinputs*
T0*(
_output_shapes
:?????????\

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:?????????"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:?????????:P L
(
_output_shapes
:?????????
 
_user_specified_nameinputs

f
J__inference_max_pooling2d_8_layer_call_and_return_conditional_losses_91477

inputs
identityข
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4????????????????????????????????????*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs


D__inference_conv2d_13_layer_call_and_return_conditional_losses_91588

inputs:
conv2d_readvariableop_resource:.
biasadd_readvariableop_resource:	
identityขBiasAdd/ReadVariableOpขConv2D/ReadVariableOp~
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:*
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????*
paddingSAME*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????Y
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:?????????j
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:?????????w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :?????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:?????????
 
_user_specified_nameinputs

ฤ
Q__inference_batch_normalization_11_layer_call_and_return_conditional_losses_91253

inputs&
readvariableop_resource:	(
readvariableop_1_resource:	7
(fusedbatchnormv3_readvariableop_resource:	9
*fusedbatchnormv3_readvariableop_1_resource:	
identityขAssignNewValueขAssignNewValue_1ขFusedBatchNormV3/ReadVariableOpข!FusedBatchNormV3/ReadVariableOp_1ขReadVariableOpขReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:*
dtype0
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype0
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype0?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,???????????????????????????:::::*
epsilon%o:*
exponential_avg_factor%
ื#<ฦ
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(ะ
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(~
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,???????????????????????????ิ
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,???????????????????????????: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,???????????????????????????
 
_user_specified_nameinputs
ข

๔
B__inference_dense_8_layer_call_and_return_conditional_losses_91697

inputs1
matmul_readvariableop_resource:	-
biasadd_readvariableop_resource:
identityขBiasAdd/ReadVariableOpขMatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????V
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:?????????`
IdentityIdentitySoftmax:softmax:0^NoOp*
T0*'
_output_shapes
:?????????w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:?????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:?????????
 
_user_specified_nameinputs
ฅ

๖
B__inference_dense_6_layer_call_and_return_conditional_losses_91649

inputs2
matmul_readvariableop_resource:
.
biasadd_readvariableop_resource:	
identityขBiasAdd/ReadVariableOpขMatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????Q
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:?????????b
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:?????????w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:?????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:?????????
 
_user_specified_nameinputs
ถ
K
/__inference_max_pooling2d_7_layer_call_fn_93157

inputs
identityุ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4????????????????????????????????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *S
fNRL
J__inference_max_pooling2d_7_layer_call_and_return_conditional_losses_91273
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs

ฤ
Q__inference_batch_normalization_12_layer_call_and_return_conditional_losses_93244

inputs&
readvariableop_resource:	(
readvariableop_1_resource:	7
(fusedbatchnormv3_readvariableop_resource:	9
*fusedbatchnormv3_readvariableop_1_resource:	
identityขAssignNewValueขAssignNewValue_1ขFusedBatchNormV3/ReadVariableOpข!FusedBatchNormV3/ReadVariableOp_1ขReadVariableOpขReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:*
dtype0
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype0
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype0?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,???????????????????????????:::::*
epsilon%o:*
exponential_avg_factor%
ื#<ฦ
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(ะ
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(~
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,???????????????????????????ิ
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,???????????????????????????: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,???????????????????????????
 
_user_specified_nameinputs

ภ
Q__inference_batch_normalization_10_layer_call_and_return_conditional_losses_91177

inputs%
readvariableop_resource:`'
readvariableop_1_resource:`6
(fusedbatchnormv3_readvariableop_resource:`8
*fusedbatchnormv3_readvariableop_1_resource:`
identityขAssignNewValueขAssignNewValue_1ขFusedBatchNormV3/ReadVariableOpข!FusedBatchNormV3/ReadVariableOp_1ขReadVariableOpขReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:`*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:`*
dtype0
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:`*
dtype0
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:`*
dtype0ึ
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????`:`:`:`:`:*
epsilon%o:*
exponential_avg_factor%
ื#<ฦ
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(ะ
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(}
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????`ิ
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????`: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????`
 
_user_specified_nameinputs
๊ซ
?
G__inference_sequential_2_layer_call_and_return_conditional_losses_92810

inputsB
(conv2d_10_conv2d_readvariableop_resource:`7
)conv2d_10_biasadd_readvariableop_resource:`<
.batch_normalization_10_readvariableop_resource:`>
0batch_normalization_10_readvariableop_1_resource:`M
?batch_normalization_10_fusedbatchnormv3_readvariableop_resource:`O
Abatch_normalization_10_fusedbatchnormv3_readvariableop_1_resource:`C
(conv2d_11_conv2d_readvariableop_resource:`8
)conv2d_11_biasadd_readvariableop_resource:	=
.batch_normalization_11_readvariableop_resource:	?
0batch_normalization_11_readvariableop_1_resource:	N
?batch_normalization_11_fusedbatchnormv3_readvariableop_resource:	P
Abatch_normalization_11_fusedbatchnormv3_readvariableop_1_resource:	D
(conv2d_12_conv2d_readvariableop_resource:8
)conv2d_12_biasadd_readvariableop_resource:	=
.batch_normalization_12_readvariableop_resource:	?
0batch_normalization_12_readvariableop_1_resource:	N
?batch_normalization_12_fusedbatchnormv3_readvariableop_resource:	P
Abatch_normalization_12_fusedbatchnormv3_readvariableop_1_resource:	D
(conv2d_13_conv2d_readvariableop_resource:8
)conv2d_13_biasadd_readvariableop_resource:	=
.batch_normalization_13_readvariableop_resource:	?
0batch_normalization_13_readvariableop_1_resource:	N
?batch_normalization_13_fusedbatchnormv3_readvariableop_resource:	P
Abatch_normalization_13_fusedbatchnormv3_readvariableop_1_resource:	D
(conv2d_14_conv2d_readvariableop_resource:8
)conv2d_14_biasadd_readvariableop_resource:	=
.batch_normalization_14_readvariableop_resource:	?
0batch_normalization_14_readvariableop_1_resource:	N
?batch_normalization_14_fusedbatchnormv3_readvariableop_resource:	P
Abatch_normalization_14_fusedbatchnormv3_readvariableop_1_resource:	:
&dense_6_matmul_readvariableop_resource:
6
'dense_6_biasadd_readvariableop_resource:	:
&dense_7_matmul_readvariableop_resource:
6
'dense_7_biasadd_readvariableop_resource:	9
&dense_8_matmul_readvariableop_resource:	5
'dense_8_biasadd_readvariableop_resource:
identityข6batch_normalization_10/FusedBatchNormV3/ReadVariableOpข8batch_normalization_10/FusedBatchNormV3/ReadVariableOp_1ข%batch_normalization_10/ReadVariableOpข'batch_normalization_10/ReadVariableOp_1ข6batch_normalization_11/FusedBatchNormV3/ReadVariableOpข8batch_normalization_11/FusedBatchNormV3/ReadVariableOp_1ข%batch_normalization_11/ReadVariableOpข'batch_normalization_11/ReadVariableOp_1ข6batch_normalization_12/FusedBatchNormV3/ReadVariableOpข8batch_normalization_12/FusedBatchNormV3/ReadVariableOp_1ข%batch_normalization_12/ReadVariableOpข'batch_normalization_12/ReadVariableOp_1ข6batch_normalization_13/FusedBatchNormV3/ReadVariableOpข8batch_normalization_13/FusedBatchNormV3/ReadVariableOp_1ข%batch_normalization_13/ReadVariableOpข'batch_normalization_13/ReadVariableOp_1ข6batch_normalization_14/FusedBatchNormV3/ReadVariableOpข8batch_normalization_14/FusedBatchNormV3/ReadVariableOp_1ข%batch_normalization_14/ReadVariableOpข'batch_normalization_14/ReadVariableOp_1ข conv2d_10/BiasAdd/ReadVariableOpขconv2d_10/Conv2D/ReadVariableOpข conv2d_11/BiasAdd/ReadVariableOpขconv2d_11/Conv2D/ReadVariableOpข conv2d_12/BiasAdd/ReadVariableOpขconv2d_12/Conv2D/ReadVariableOpข conv2d_13/BiasAdd/ReadVariableOpขconv2d_13/Conv2D/ReadVariableOpข conv2d_14/BiasAdd/ReadVariableOpขconv2d_14/Conv2D/ReadVariableOpขdense_6/BiasAdd/ReadVariableOpขdense_6/MatMul/ReadVariableOpขdense_7/BiasAdd/ReadVariableOpขdense_7/MatMul/ReadVariableOpขdense_8/BiasAdd/ReadVariableOpขdense_8/MatMul/ReadVariableOpW
rescaling_2/Cast/xConst*
_output_shapes
: *
dtype0*
valueB
 *;Y
rescaling_2/Cast_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *    w
rescaling_2/mulMulinputsrescaling_2/Cast/x:output:0*
T0*1
_output_shapes
:?????????เ
rescaling_2/addAddV2rescaling_2/mul:z:0rescaling_2/Cast_1/x:output:0*
T0*1
_output_shapes
:?????????เ
conv2d_10/Conv2D/ReadVariableOpReadVariableOp(conv2d_10_conv2d_readvariableop_resource*&
_output_shapes
:`*
dtype0ผ
conv2d_10/Conv2DConv2Drescaling_2/add:z:0'conv2d_10/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????v`*
paddingVALID*
strides

 conv2d_10/BiasAdd/ReadVariableOpReadVariableOp)conv2d_10_biasadd_readvariableop_resource*
_output_shapes
:`*
dtype0
conv2d_10/BiasAddBiasAddconv2d_10/Conv2D:output:0(conv2d_10/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????v`m
conv2d_10/ReluReluconv2d_10/BiasAdd:output:0*
T0*0
_output_shapes
:?????????v`
%batch_normalization_10/ReadVariableOpReadVariableOp.batch_normalization_10_readvariableop_resource*
_output_shapes
:`*
dtype0
'batch_normalization_10/ReadVariableOp_1ReadVariableOp0batch_normalization_10_readvariableop_1_resource*
_output_shapes
:`*
dtype0ฒ
6batch_normalization_10/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_10_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:`*
dtype0ถ
8batch_normalization_10/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_10_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:`*
dtype0ภ
'batch_normalization_10/FusedBatchNormV3FusedBatchNormV3conv2d_10/Relu:activations:0-batch_normalization_10/ReadVariableOp:value:0/batch_normalization_10/ReadVariableOp_1:value:0>batch_normalization_10/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_10/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*L
_output_shapes:
8:?????????v`:`:`:`:`:*
epsilon%o:*
is_training( ผ
max_pooling2d_6/MaxPoolMaxPool+batch_normalization_10/FusedBatchNormV3:y:0*/
_output_shapes
:?????????:N`*
ksize
*
paddingVALID*
strides

conv2d_11/Conv2D/ReadVariableOpReadVariableOp(conv2d_11_conv2d_readvariableop_resource*'
_output_shapes
:`*
dtype0ศ
conv2d_11/Conv2DConv2D max_pooling2d_6/MaxPool:output:0'conv2d_11/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????*
paddingSAME*
strides

 conv2d_11/BiasAdd/ReadVariableOpReadVariableOp)conv2d_11_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0
conv2d_11/BiasAddBiasAddconv2d_11/Conv2D:output:0(conv2d_11/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????m
conv2d_11/ReluReluconv2d_11/BiasAdd:output:0*
T0*0
_output_shapes
:?????????
%batch_normalization_11/ReadVariableOpReadVariableOp.batch_normalization_11_readvariableop_resource*
_output_shapes	
:*
dtype0
'batch_normalization_11/ReadVariableOp_1ReadVariableOp0batch_normalization_11_readvariableop_1_resource*
_output_shapes	
:*
dtype0ณ
6batch_normalization_11/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_11_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype0ท
8batch_normalization_11/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_11_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype0ฤ
'batch_normalization_11/FusedBatchNormV3FusedBatchNormV3conv2d_11/Relu:activations:0-batch_normalization_11/ReadVariableOp:value:0/batch_normalization_11/ReadVariableOp_1:value:0>batch_normalization_11/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_11/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:?????????:::::*
epsilon%o:*
is_training( ฝ
max_pooling2d_7/MaxPoolMaxPool+batch_normalization_11/FusedBatchNormV3:y:0*0
_output_shapes
:?????????	*
ksize
*
paddingVALID*
strides

conv2d_12/Conv2D/ReadVariableOpReadVariableOp(conv2d_12_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0ศ
conv2d_12/Conv2DConv2D max_pooling2d_7/MaxPool:output:0'conv2d_12/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????*
paddingSAME*
strides

 conv2d_12/BiasAdd/ReadVariableOpReadVariableOp)conv2d_12_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0
conv2d_12/BiasAddBiasAddconv2d_12/Conv2D:output:0(conv2d_12/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????m
conv2d_12/ReluReluconv2d_12/BiasAdd:output:0*
T0*0
_output_shapes
:?????????
%batch_normalization_12/ReadVariableOpReadVariableOp.batch_normalization_12_readvariableop_resource*
_output_shapes	
:*
dtype0
'batch_normalization_12/ReadVariableOp_1ReadVariableOp0batch_normalization_12_readvariableop_1_resource*
_output_shapes	
:*
dtype0ณ
6batch_normalization_12/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_12_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype0ท
8batch_normalization_12/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_12_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype0ฤ
'batch_normalization_12/FusedBatchNormV3FusedBatchNormV3conv2d_12/Relu:activations:0-batch_normalization_12/ReadVariableOp:value:0/batch_normalization_12/ReadVariableOp_1:value:0>batch_normalization_12/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_12/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:?????????:::::*
epsilon%o:*
is_training( 
conv2d_13/Conv2D/ReadVariableOpReadVariableOp(conv2d_13_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0ำ
conv2d_13/Conv2DConv2D+batch_normalization_12/FusedBatchNormV3:y:0'conv2d_13/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????*
paddingSAME*
strides

 conv2d_13/BiasAdd/ReadVariableOpReadVariableOp)conv2d_13_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0
conv2d_13/BiasAddBiasAddconv2d_13/Conv2D:output:0(conv2d_13/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????m
conv2d_13/ReluReluconv2d_13/BiasAdd:output:0*
T0*0
_output_shapes
:?????????
%batch_normalization_13/ReadVariableOpReadVariableOp.batch_normalization_13_readvariableop_resource*
_output_shapes	
:*
dtype0
'batch_normalization_13/ReadVariableOp_1ReadVariableOp0batch_normalization_13_readvariableop_1_resource*
_output_shapes	
:*
dtype0ณ
6batch_normalization_13/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_13_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype0ท
8batch_normalization_13/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_13_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype0ฤ
'batch_normalization_13/FusedBatchNormV3FusedBatchNormV3conv2d_13/Relu:activations:0-batch_normalization_13/ReadVariableOp:value:0/batch_normalization_13/ReadVariableOp_1:value:0>batch_normalization_13/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_13/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:?????????:::::*
epsilon%o:*
is_training( 
conv2d_14/Conv2D/ReadVariableOpReadVariableOp(conv2d_14_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0ำ
conv2d_14/Conv2DConv2D+batch_normalization_13/FusedBatchNormV3:y:0'conv2d_14/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????*
paddingSAME*
strides

 conv2d_14/BiasAdd/ReadVariableOpReadVariableOp)conv2d_14_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0
conv2d_14/BiasAddBiasAddconv2d_14/Conv2D:output:0(conv2d_14/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????m
conv2d_14/ReluReluconv2d_14/BiasAdd:output:0*
T0*0
_output_shapes
:?????????
%batch_normalization_14/ReadVariableOpReadVariableOp.batch_normalization_14_readvariableop_resource*
_output_shapes	
:*
dtype0
'batch_normalization_14/ReadVariableOp_1ReadVariableOp0batch_normalization_14_readvariableop_1_resource*
_output_shapes	
:*
dtype0ณ
6batch_normalization_14/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_14_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype0ท
8batch_normalization_14/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_14_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype0ฤ
'batch_normalization_14/FusedBatchNormV3FusedBatchNormV3conv2d_14/Relu:activations:0-batch_normalization_14/ReadVariableOp:value:0/batch_normalization_14/ReadVariableOp_1:value:0>batch_normalization_14/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_14/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:?????????:::::*
epsilon%o:*
is_training( ฝ
max_pooling2d_8/MaxPoolMaxPool+batch_normalization_14/FusedBatchNormV3:y:0*0
_output_shapes
:?????????*
ksize
*
paddingVALID*
strides
`
flatten_2/ConstConst*
_output_shapes
:*
dtype0*
valueB"????   
flatten_2/ReshapeReshape max_pooling2d_8/MaxPool:output:0flatten_2/Const:output:0*
T0*(
_output_shapes
:?????????
dense_6/MatMul/ReadVariableOpReadVariableOp&dense_6_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype0
dense_6/MatMulMatMulflatten_2/Reshape:output:0%dense_6/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????
dense_6/BiasAdd/ReadVariableOpReadVariableOp'dense_6_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0
dense_6/BiasAddBiasAdddense_6/MatMul:product:0&dense_6/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????a
dense_6/ReluReludense_6/BiasAdd:output:0*
T0*(
_output_shapes
:?????????m
dropout_4/IdentityIdentitydense_6/Relu:activations:0*
T0*(
_output_shapes
:?????????
dense_7/MatMul/ReadVariableOpReadVariableOp&dense_7_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype0
dense_7/MatMulMatMuldropout_4/Identity:output:0%dense_7/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????
dense_7/BiasAdd/ReadVariableOpReadVariableOp'dense_7_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0
dense_7/BiasAddBiasAdddense_7/MatMul:product:0&dense_7/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????a
dense_7/ReluReludense_7/BiasAdd:output:0*
T0*(
_output_shapes
:?????????m
dropout_5/IdentityIdentitydense_7/Relu:activations:0*
T0*(
_output_shapes
:?????????
dense_8/MatMul/ReadVariableOpReadVariableOp&dense_8_matmul_readvariableop_resource*
_output_shapes
:	*
dtype0
dense_8/MatMulMatMuldropout_5/Identity:output:0%dense_8/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
dense_8/BiasAdd/ReadVariableOpReadVariableOp'dense_8_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense_8/BiasAddBiasAdddense_8/MatMul:product:0&dense_8/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????f
dense_8/SoftmaxSoftmaxdense_8/BiasAdd:output:0*
T0*'
_output_shapes
:?????????h
IdentityIdentitydense_8/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:?????????ภ
NoOpNoOp7^batch_normalization_10/FusedBatchNormV3/ReadVariableOp9^batch_normalization_10/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_10/ReadVariableOp(^batch_normalization_10/ReadVariableOp_17^batch_normalization_11/FusedBatchNormV3/ReadVariableOp9^batch_normalization_11/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_11/ReadVariableOp(^batch_normalization_11/ReadVariableOp_17^batch_normalization_12/FusedBatchNormV3/ReadVariableOp9^batch_normalization_12/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_12/ReadVariableOp(^batch_normalization_12/ReadVariableOp_17^batch_normalization_13/FusedBatchNormV3/ReadVariableOp9^batch_normalization_13/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_13/ReadVariableOp(^batch_normalization_13/ReadVariableOp_17^batch_normalization_14/FusedBatchNormV3/ReadVariableOp9^batch_normalization_14/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_14/ReadVariableOp(^batch_normalization_14/ReadVariableOp_1!^conv2d_10/BiasAdd/ReadVariableOp ^conv2d_10/Conv2D/ReadVariableOp!^conv2d_11/BiasAdd/ReadVariableOp ^conv2d_11/Conv2D/ReadVariableOp!^conv2d_12/BiasAdd/ReadVariableOp ^conv2d_12/Conv2D/ReadVariableOp!^conv2d_13/BiasAdd/ReadVariableOp ^conv2d_13/Conv2D/ReadVariableOp!^conv2d_14/BiasAdd/ReadVariableOp ^conv2d_14/Conv2D/ReadVariableOp^dense_6/BiasAdd/ReadVariableOp^dense_6/MatMul/ReadVariableOp^dense_7/BiasAdd/ReadVariableOp^dense_7/MatMul/ReadVariableOp^dense_8/BiasAdd/ReadVariableOp^dense_8/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*x
_input_shapesg
e:?????????เ: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2p
6batch_normalization_10/FusedBatchNormV3/ReadVariableOp6batch_normalization_10/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_10/FusedBatchNormV3/ReadVariableOp_18batch_normalization_10/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_10/ReadVariableOp%batch_normalization_10/ReadVariableOp2R
'batch_normalization_10/ReadVariableOp_1'batch_normalization_10/ReadVariableOp_12p
6batch_normalization_11/FusedBatchNormV3/ReadVariableOp6batch_normalization_11/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_11/FusedBatchNormV3/ReadVariableOp_18batch_normalization_11/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_11/ReadVariableOp%batch_normalization_11/ReadVariableOp2R
'batch_normalization_11/ReadVariableOp_1'batch_normalization_11/ReadVariableOp_12p
6batch_normalization_12/FusedBatchNormV3/ReadVariableOp6batch_normalization_12/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_12/FusedBatchNormV3/ReadVariableOp_18batch_normalization_12/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_12/ReadVariableOp%batch_normalization_12/ReadVariableOp2R
'batch_normalization_12/ReadVariableOp_1'batch_normalization_12/ReadVariableOp_12p
6batch_normalization_13/FusedBatchNormV3/ReadVariableOp6batch_normalization_13/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_13/FusedBatchNormV3/ReadVariableOp_18batch_normalization_13/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_13/ReadVariableOp%batch_normalization_13/ReadVariableOp2R
'batch_normalization_13/ReadVariableOp_1'batch_normalization_13/ReadVariableOp_12p
6batch_normalization_14/FusedBatchNormV3/ReadVariableOp6batch_normalization_14/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_14/FusedBatchNormV3/ReadVariableOp_18batch_normalization_14/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_14/ReadVariableOp%batch_normalization_14/ReadVariableOp2R
'batch_normalization_14/ReadVariableOp_1'batch_normalization_14/ReadVariableOp_12D
 conv2d_10/BiasAdd/ReadVariableOp conv2d_10/BiasAdd/ReadVariableOp2B
conv2d_10/Conv2D/ReadVariableOpconv2d_10/Conv2D/ReadVariableOp2D
 conv2d_11/BiasAdd/ReadVariableOp conv2d_11/BiasAdd/ReadVariableOp2B
conv2d_11/Conv2D/ReadVariableOpconv2d_11/Conv2D/ReadVariableOp2D
 conv2d_12/BiasAdd/ReadVariableOp conv2d_12/BiasAdd/ReadVariableOp2B
conv2d_12/Conv2D/ReadVariableOpconv2d_12/Conv2D/ReadVariableOp2D
 conv2d_13/BiasAdd/ReadVariableOp conv2d_13/BiasAdd/ReadVariableOp2B
conv2d_13/Conv2D/ReadVariableOpconv2d_13/Conv2D/ReadVariableOp2D
 conv2d_14/BiasAdd/ReadVariableOp conv2d_14/BiasAdd/ReadVariableOp2B
conv2d_14/Conv2D/ReadVariableOpconv2d_14/Conv2D/ReadVariableOp2@
dense_6/BiasAdd/ReadVariableOpdense_6/BiasAdd/ReadVariableOp2>
dense_6/MatMul/ReadVariableOpdense_6/MatMul/ReadVariableOp2@
dense_7/BiasAdd/ReadVariableOpdense_7/BiasAdd/ReadVariableOp2>
dense_7/MatMul/ReadVariableOpdense_7/MatMul/ReadVariableOp2@
dense_8/BiasAdd/ReadVariableOpdense_8/BiasAdd/ReadVariableOp2>
dense_8/MatMul/ReadVariableOpdense_8/MatMul/ReadVariableOp:Y U
1
_output_shapes
:?????????เ
 
_user_specified_nameinputs
	
ี
6__inference_batch_normalization_13_layer_call_fn_93290

inputs
unknown:	
	unknown_0:	
	unknown_1:	
	unknown_2:	
identityขStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,???????????????????????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Z
fURS
Q__inference_batch_normalization_13_layer_call_and_return_conditional_losses_91393
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*B
_output_shapes0
.:,???????????????????????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,???????????????????????????: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,???????????????????????????
 
_user_specified_nameinputs"ฟL
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*ศ
serving_defaultด
Y
rescaling_2_inputD
#serving_default_rescaling_2_input:0?????????เ;
dense_80
StatefulPartitionedCall:0?????????tensorflow/serving/predict:?
๎
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer-3
layer_with_weights-2
layer-4
layer_with_weights-3
layer-5
layer-6
layer_with_weights-4
layer-7
	layer_with_weights-5
	layer-8

layer_with_weights-6

layer-9
layer_with_weights-7
layer-10
layer_with_weights-8
layer-11
layer_with_weights-9
layer-12
layer-13
layer-14
layer_with_weights-10
layer-15
layer-16
layer_with_weights-11
layer-17
layer-18
layer_with_weights-12
layer-19
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature
	optimizer

signatures
#_self_saveable_object_factories"
_tf_keras_sequential
ส
	variables
 trainable_variables
!regularization_losses
"	keras_api
#__call__
*$&call_and_return_all_conditional_losses
#%_self_saveable_object_factories"
_tf_keras_layer

&	variables
'trainable_variables
(regularization_losses
)	keras_api
*__call__
*+&call_and_return_all_conditional_losses

,kernel
-bias
#._self_saveable_object_factories
 /_jit_compiled_convolution_op"
_tf_keras_layer

0	variables
1trainable_variables
2regularization_losses
3	keras_api
4__call__
*5&call_and_return_all_conditional_losses
6axis
	7gamma
8beta
9moving_mean
:moving_variance
#;_self_saveable_object_factories"
_tf_keras_layer
ส
<	variables
=trainable_variables
>regularization_losses
?	keras_api
@__call__
*A&call_and_return_all_conditional_losses
#B_self_saveable_object_factories"
_tf_keras_layer

C	variables
Dtrainable_variables
Eregularization_losses
F	keras_api
G__call__
*H&call_and_return_all_conditional_losses

Ikernel
Jbias
#K_self_saveable_object_factories
 L_jit_compiled_convolution_op"
_tf_keras_layer

M	variables
Ntrainable_variables
Oregularization_losses
P	keras_api
Q__call__
*R&call_and_return_all_conditional_losses
Saxis
	Tgamma
Ubeta
Vmoving_mean
Wmoving_variance
#X_self_saveable_object_factories"
_tf_keras_layer
ส
Y	variables
Ztrainable_variables
[regularization_losses
\	keras_api
]__call__
*^&call_and_return_all_conditional_losses
#__self_saveable_object_factories"
_tf_keras_layer

`	variables
atrainable_variables
bregularization_losses
c	keras_api
d__call__
*e&call_and_return_all_conditional_losses

fkernel
gbias
#h_self_saveable_object_factories
 i_jit_compiled_convolution_op"
_tf_keras_layer

j	variables
ktrainable_variables
lregularization_losses
m	keras_api
n__call__
*o&call_and_return_all_conditional_losses
paxis
	qgamma
rbeta
smoving_mean
tmoving_variance
#u_self_saveable_object_factories"
_tf_keras_layer

v	variables
wtrainable_variables
xregularization_losses
y	keras_api
z__call__
*{&call_and_return_all_conditional_losses

|kernel
}bias
#~_self_saveable_object_factories
 _jit_compiled_convolution_op"
_tf_keras_layer

	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses
	axis

gamma
	beta
moving_mean
moving_variance
$_self_saveable_object_factories"
_tf_keras_layer

	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses
kernel
	bias
$_self_saveable_object_factories
!_jit_compiled_convolution_op"
_tf_keras_layer

	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses
	axis

gamma
	beta
moving_mean
?moving_variance
$ก_self_saveable_object_factories"
_tf_keras_layer
ั
ข	variables
ฃtrainable_variables
คregularization_losses
ฅ	keras_api
ฆ__call__
+ง&call_and_return_all_conditional_losses
$จ_self_saveable_object_factories"
_tf_keras_layer
ั
ฉ	variables
ชtrainable_variables
ซregularization_losses
ฌ	keras_api
ญ__call__
+ฎ&call_and_return_all_conditional_losses
$ฏ_self_saveable_object_factories"
_tf_keras_layer
้
ฐ	variables
ฑtrainable_variables
ฒregularization_losses
ณ	keras_api
ด__call__
+ต&call_and_return_all_conditional_losses
ถkernel
	ทbias
$ธ_self_saveable_object_factories"
_tf_keras_layer
้
น	variables
บtrainable_variables
ปregularization_losses
ผ	keras_api
ฝ__call__
+พ&call_and_return_all_conditional_losses
ฟ_random_generator
$ภ_self_saveable_object_factories"
_tf_keras_layer
้
ม	variables
ยtrainable_variables
รregularization_losses
ฤ	keras_api
ล__call__
+ฦ&call_and_return_all_conditional_losses
วkernel
	ศbias
$ษ_self_saveable_object_factories"
_tf_keras_layer
้
ส	variables
หtrainable_variables
ฬregularization_losses
อ	keras_api
ฮ__call__
+ฯ&call_and_return_all_conditional_losses
ะ_random_generator
$ั_self_saveable_object_factories"
_tf_keras_layer
้
า	variables
ำtrainable_variables
ิregularization_losses
ี	keras_api
ึ__call__
+ื&call_and_return_all_conditional_losses
ุkernel
	ูbias
$ฺ_self_saveable_object_factories"
_tf_keras_layer
ฦ
,0
-1
72
83
94
:5
I6
J7
T8
U9
V10
W11
f12
g13
q14
r15
s16
t17
|18
}19
20
21
22
23
24
25
26
27
28
?29
ถ30
ท31
ว32
ศ33
ุ34
ู35"
trackable_list_wrapper
๒
,0
-1
72
83
I4
J5
T6
U7
f8
g9
q10
r11
|12
}13
14
15
16
17
18
19
ถ20
ท21
ว22
ศ23
ุ24
ู25"
trackable_list_wrapper
 "
trackable_list_wrapper
ฯ
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
฿layer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
๎
เtrace_0
แtrace_1
โtrace_2
ใtrace_32๛
,__inference_sequential_2_layer_call_fn_91779
,__inference_sequential_2_layer_call_fn_92592
,__inference_sequential_2_layer_call_fn_92669
,__inference_sequential_2_layer_call_fn_92244ภ
ทฒณ
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsช 
annotationsช *
 zเtrace_0zแtrace_1zโtrace_2zใtrace_3
ฺ
ไtrace_0
ๅtrace_1
ๆtrace_2
็trace_32็
G__inference_sequential_2_layer_call_and_return_conditional_losses_92810
G__inference_sequential_2_layer_call_and_return_conditional_losses_92965
G__inference_sequential_2_layer_call_and_return_conditional_losses_92340
G__inference_sequential_2_layer_call_and_return_conditional_losses_92436ภ
ทฒณ
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsช 
annotationsช *
 zไtrace_0zๅtrace_1zๆtrace_2z็trace_3
ีBา
 __inference__wrapped_model_91124rescaling_2_input"
ฒ
FullArgSpec
args 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsช *
 
"
	optimizer
-
่serving_default"
signature_map
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
ฒ
้non_trainable_variables
๊layers
๋metrics
 ์layer_regularization_losses
ํlayer_metrics
	variables
 trainable_variables
!regularization_losses
#__call__
*$&call_and_return_all_conditional_losses
&$"call_and_return_conditional_losses"
_generic_user_object
๑
๎trace_02า
+__inference_rescaling_2_layer_call_fn_92970ข
ฒ
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsช *
 z๎trace_0

๏trace_02ํ
F__inference_rescaling_2_layer_call_and_return_conditional_losses_92978ข
ฒ
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsช *
 z๏trace_0
 "
trackable_dict_wrapper
.
,0
-1"
trackable_list_wrapper
.
,0
-1"
trackable_list_wrapper
 "
trackable_list_wrapper
ฒ
๐non_trainable_variables
๑layers
๒metrics
 ๓layer_regularization_losses
๔layer_metrics
&	variables
'trainable_variables
(regularization_losses
*__call__
*+&call_and_return_all_conditional_losses
&+"call_and_return_conditional_losses"
_generic_user_object
๏
๕trace_02ะ
)__inference_conv2d_10_layer_call_fn_92987ข
ฒ
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsช *
 z๕trace_0

๖trace_02๋
D__inference_conv2d_10_layer_call_and_return_conditional_losses_92998ข
ฒ
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsช *
 z๖trace_0
*:(`2conv2d_10/kernel
:`2conv2d_10/bias
 "
trackable_dict_wrapper
ด2ฑฎ
ฃฒ
FullArgSpec'
args
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsช *
 0
<
70
81
92
:3"
trackable_list_wrapper
.
70
81"
trackable_list_wrapper
 "
trackable_list_wrapper
ฒ
๗non_trainable_variables
๘layers
๙metrics
 ๚layer_regularization_losses
๛layer_metrics
0	variables
1trainable_variables
2regularization_losses
4__call__
*5&call_and_return_all_conditional_losses
&5"call_and_return_conditional_losses"
_generic_user_object
โ
?trace_0
?trace_12ง
6__inference_batch_normalization_10_layer_call_fn_93011
6__inference_batch_normalization_10_layer_call_fn_93024ด
ซฒง
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsช 
annotationsช *
 z?trace_0z?trace_1

?trace_0
?trace_12?
Q__inference_batch_normalization_10_layer_call_and_return_conditional_losses_93042
Q__inference_batch_normalization_10_layer_call_and_return_conditional_losses_93060ด
ซฒง
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsช 
annotationsช *
 z?trace_0z?trace_1
 "
trackable_list_wrapper
*:(`2batch_normalization_10/gamma
):'`2batch_normalization_10/beta
2:0` (2"batch_normalization_10/moving_mean
6:4` (2&batch_normalization_10/moving_variance
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
ฒ
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
<	variables
=trainable_variables
>regularization_losses
@__call__
*A&call_and_return_all_conditional_losses
&A"call_and_return_conditional_losses"
_generic_user_object
๕
trace_02ึ
/__inference_max_pooling2d_6_layer_call_fn_93065ข
ฒ
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsช *
 ztrace_0

trace_02๑
J__inference_max_pooling2d_6_layer_call_and_return_conditional_losses_93070ข
ฒ
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsช *
 ztrace_0
 "
trackable_dict_wrapper
.
I0
J1"
trackable_list_wrapper
.
I0
J1"
trackable_list_wrapper
 "
trackable_list_wrapper
ฒ
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
C	variables
Dtrainable_variables
Eregularization_losses
G__call__
*H&call_and_return_all_conditional_losses
&H"call_and_return_conditional_losses"
_generic_user_object
๏
trace_02ะ
)__inference_conv2d_11_layer_call_fn_93079ข
ฒ
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsช *
 ztrace_0

trace_02๋
D__inference_conv2d_11_layer_call_and_return_conditional_losses_93090ข
ฒ
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsช *
 ztrace_0
+:)`2conv2d_11/kernel
:2conv2d_11/bias
 "
trackable_dict_wrapper
ด2ฑฎ
ฃฒ
FullArgSpec'
args
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsช *
 0
<
T0
U1
V2
W3"
trackable_list_wrapper
.
T0
U1"
trackable_list_wrapper
 "
trackable_list_wrapper
ฒ
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
M	variables
Ntrainable_variables
Oregularization_losses
Q__call__
*R&call_and_return_all_conditional_losses
&R"call_and_return_conditional_losses"
_generic_user_object
โ
trace_0
trace_12ง
6__inference_batch_normalization_11_layer_call_fn_93103
6__inference_batch_normalization_11_layer_call_fn_93116ด
ซฒง
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsช 
annotationsช *
 ztrace_0ztrace_1

trace_0
trace_12?
Q__inference_batch_normalization_11_layer_call_and_return_conditional_losses_93134
Q__inference_batch_normalization_11_layer_call_and_return_conditional_losses_93152ด
ซฒง
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsช 
annotationsช *
 ztrace_0ztrace_1
 "
trackable_list_wrapper
+:)2batch_normalization_11/gamma
*:(2batch_normalization_11/beta
3:1 (2"batch_normalization_11/moving_mean
7:5 (2&batch_normalization_11/moving_variance
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
ฒ
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
Y	variables
Ztrainable_variables
[regularization_losses
]__call__
*^&call_and_return_all_conditional_losses
&^"call_and_return_conditional_losses"
_generic_user_object
๕
trace_02ึ
/__inference_max_pooling2d_7_layer_call_fn_93157ข
ฒ
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsช *
 ztrace_0

trace_02๑
J__inference_max_pooling2d_7_layer_call_and_return_conditional_losses_93162ข
ฒ
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsช *
 ztrace_0
 "
trackable_dict_wrapper
.
f0
g1"
trackable_list_wrapper
.
f0
g1"
trackable_list_wrapper
 "
trackable_list_wrapper
ฒ
non_trainable_variables
layers
?metrics
 กlayer_regularization_losses
ขlayer_metrics
`	variables
atrainable_variables
bregularization_losses
d__call__
*e&call_and_return_all_conditional_losses
&e"call_and_return_conditional_losses"
_generic_user_object
๏
ฃtrace_02ะ
)__inference_conv2d_12_layer_call_fn_93171ข
ฒ
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsช *
 zฃtrace_0

คtrace_02๋
D__inference_conv2d_12_layer_call_and_return_conditional_losses_93182ข
ฒ
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsช *
 zคtrace_0
,:*2conv2d_12/kernel
:2conv2d_12/bias
 "
trackable_dict_wrapper
ด2ฑฎ
ฃฒ
FullArgSpec'
args
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsช *
 0
<
q0
r1
s2
t3"
trackable_list_wrapper
.
q0
r1"
trackable_list_wrapper
 "
trackable_list_wrapper
ฒ
ฅnon_trainable_variables
ฆlayers
งmetrics
 จlayer_regularization_losses
ฉlayer_metrics
j	variables
ktrainable_variables
lregularization_losses
n__call__
*o&call_and_return_all_conditional_losses
&o"call_and_return_conditional_losses"
_generic_user_object
โ
ชtrace_0
ซtrace_12ง
6__inference_batch_normalization_12_layer_call_fn_93195
6__inference_batch_normalization_12_layer_call_fn_93208ด
ซฒง
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsช 
annotationsช *
 zชtrace_0zซtrace_1

ฌtrace_0
ญtrace_12?
Q__inference_batch_normalization_12_layer_call_and_return_conditional_losses_93226
Q__inference_batch_normalization_12_layer_call_and_return_conditional_losses_93244ด
ซฒง
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsช 
annotationsช *
 zฌtrace_0zญtrace_1
 "
trackable_list_wrapper
+:)2batch_normalization_12/gamma
*:(2batch_normalization_12/beta
3:1 (2"batch_normalization_12/moving_mean
7:5 (2&batch_normalization_12/moving_variance
 "
trackable_dict_wrapper
.
|0
}1"
trackable_list_wrapper
.
|0
}1"
trackable_list_wrapper
 "
trackable_list_wrapper
ฒ
ฎnon_trainable_variables
ฏlayers
ฐmetrics
 ฑlayer_regularization_losses
ฒlayer_metrics
v	variables
wtrainable_variables
xregularization_losses
z__call__
*{&call_and_return_all_conditional_losses
&{"call_and_return_conditional_losses"
_generic_user_object
๏
ณtrace_02ะ
)__inference_conv2d_13_layer_call_fn_93253ข
ฒ
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsช *
 zณtrace_0

ดtrace_02๋
D__inference_conv2d_13_layer_call_and_return_conditional_losses_93264ข
ฒ
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsช *
 zดtrace_0
,:*2conv2d_13/kernel
:2conv2d_13/bias
 "
trackable_dict_wrapper
ด2ฑฎ
ฃฒ
FullArgSpec'
args
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsช *
 0
@
0
1
2
3"
trackable_list_wrapper
0
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
ธ
ตnon_trainable_variables
ถlayers
ทmetrics
 ธlayer_regularization_losses
นlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
โ
บtrace_0
ปtrace_12ง
6__inference_batch_normalization_13_layer_call_fn_93277
6__inference_batch_normalization_13_layer_call_fn_93290ด
ซฒง
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsช 
annotationsช *
 zบtrace_0zปtrace_1

ผtrace_0
ฝtrace_12?
Q__inference_batch_normalization_13_layer_call_and_return_conditional_losses_93308
Q__inference_batch_normalization_13_layer_call_and_return_conditional_losses_93326ด
ซฒง
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsช 
annotationsช *
 zผtrace_0zฝtrace_1
 "
trackable_list_wrapper
+:)2batch_normalization_13/gamma
*:(2batch_normalization_13/beta
3:1 (2"batch_normalization_13/moving_mean
7:5 (2&batch_normalization_13/moving_variance
 "
trackable_dict_wrapper
0
0
1"
trackable_list_wrapper
0
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
ธ
พnon_trainable_variables
ฟlayers
ภmetrics
 มlayer_regularization_losses
ยlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
๏
รtrace_02ะ
)__inference_conv2d_14_layer_call_fn_93335ข
ฒ
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsช *
 zรtrace_0

ฤtrace_02๋
D__inference_conv2d_14_layer_call_and_return_conditional_losses_93346ข
ฒ
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsช *
 zฤtrace_0
,:*2conv2d_14/kernel
:2conv2d_14/bias
 "
trackable_dict_wrapper
ด2ฑฎ
ฃฒ
FullArgSpec'
args
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsช *
 0
@
0
1
2
?3"
trackable_list_wrapper
0
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
ธ
ลnon_trainable_variables
ฦlayers
วmetrics
 ศlayer_regularization_losses
ษlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
โ
สtrace_0
หtrace_12ง
6__inference_batch_normalization_14_layer_call_fn_93359
6__inference_batch_normalization_14_layer_call_fn_93372ด
ซฒง
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsช 
annotationsช *
 zสtrace_0zหtrace_1

ฬtrace_0
อtrace_12?
Q__inference_batch_normalization_14_layer_call_and_return_conditional_losses_93390
Q__inference_batch_normalization_14_layer_call_and_return_conditional_losses_93408ด
ซฒง
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsช 
annotationsช *
 zฬtrace_0zอtrace_1
 "
trackable_list_wrapper
+:)2batch_normalization_14/gamma
*:(2batch_normalization_14/beta
3:1 (2"batch_normalization_14/moving_mean
7:5 (2&batch_normalization_14/moving_variance
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
ธ
ฮnon_trainable_variables
ฯlayers
ะmetrics
 ัlayer_regularization_losses
าlayer_metrics
ข	variables
ฃtrainable_variables
คregularization_losses
ฆ__call__
+ง&call_and_return_all_conditional_losses
'ง"call_and_return_conditional_losses"
_generic_user_object
๕
ำtrace_02ึ
/__inference_max_pooling2d_8_layer_call_fn_93413ข
ฒ
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsช *
 zำtrace_0

ิtrace_02๑
J__inference_max_pooling2d_8_layer_call_and_return_conditional_losses_93418ข
ฒ
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsช *
 zิtrace_0
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
ธ
ีnon_trainable_variables
ึlayers
ืmetrics
 ุlayer_regularization_losses
ูlayer_metrics
ฉ	variables
ชtrainable_variables
ซregularization_losses
ญ__call__
+ฎ&call_and_return_all_conditional_losses
'ฎ"call_and_return_conditional_losses"
_generic_user_object
๏
ฺtrace_02ะ
)__inference_flatten_2_layer_call_fn_93423ข
ฒ
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsช *
 zฺtrace_0

?trace_02๋
D__inference_flatten_2_layer_call_and_return_conditional_losses_93429ข
ฒ
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsช *
 z?trace_0
 "
trackable_dict_wrapper
0
ถ0
ท1"
trackable_list_wrapper
0
ถ0
ท1"
trackable_list_wrapper
 "
trackable_list_wrapper
ธ
?non_trainable_variables
?layers
?metrics
 ฿layer_regularization_losses
เlayer_metrics
ฐ	variables
ฑtrainable_variables
ฒregularization_losses
ด__call__
+ต&call_and_return_all_conditional_losses
'ต"call_and_return_conditional_losses"
_generic_user_object
ํ
แtrace_02ฮ
'__inference_dense_6_layer_call_fn_93438ข
ฒ
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsช *
 zแtrace_0

โtrace_02้
B__inference_dense_6_layer_call_and_return_conditional_losses_93449ข
ฒ
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsช *
 zโtrace_0
": 
2dense_6/kernel
:2dense_6/bias
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
ธ
ใnon_trainable_variables
ไlayers
ๅmetrics
 ๆlayer_regularization_losses
็layer_metrics
น	variables
บtrainable_variables
ปregularization_losses
ฝ__call__
+พ&call_and_return_all_conditional_losses
'พ"call_and_return_conditional_losses"
_generic_user_object
ศ
่trace_0
้trace_12
)__inference_dropout_4_layer_call_fn_93454
)__inference_dropout_4_layer_call_fn_93459ด
ซฒง
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsช 
annotationsช *
 z่trace_0z้trace_1
?
๊trace_0
๋trace_12ร
D__inference_dropout_4_layer_call_and_return_conditional_losses_93464
D__inference_dropout_4_layer_call_and_return_conditional_losses_93476ด
ซฒง
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsช 
annotationsช *
 z๊trace_0z๋trace_1
"
_generic_user_object
 "
trackable_dict_wrapper
0
ว0
ศ1"
trackable_list_wrapper
0
ว0
ศ1"
trackable_list_wrapper
 "
trackable_list_wrapper
ธ
์non_trainable_variables
ํlayers
๎metrics
 ๏layer_regularization_losses
๐layer_metrics
ม	variables
ยtrainable_variables
รregularization_losses
ล__call__
+ฦ&call_and_return_all_conditional_losses
'ฦ"call_and_return_conditional_losses"
_generic_user_object
ํ
๑trace_02ฮ
'__inference_dense_7_layer_call_fn_93485ข
ฒ
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsช *
 z๑trace_0

๒trace_02้
B__inference_dense_7_layer_call_and_return_conditional_losses_93496ข
ฒ
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsช *
 z๒trace_0
": 
2dense_7/kernel
:2dense_7/bias
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
ธ
๓non_trainable_variables
๔layers
๕metrics
 ๖layer_regularization_losses
๗layer_metrics
ส	variables
หtrainable_variables
ฬregularization_losses
ฮ__call__
+ฯ&call_and_return_all_conditional_losses
'ฯ"call_and_return_conditional_losses"
_generic_user_object
ศ
๘trace_0
๙trace_12
)__inference_dropout_5_layer_call_fn_93501
)__inference_dropout_5_layer_call_fn_93506ด
ซฒง
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsช 
annotationsช *
 z๘trace_0z๙trace_1
?
๚trace_0
๛trace_12ร
D__inference_dropout_5_layer_call_and_return_conditional_losses_93511
D__inference_dropout_5_layer_call_and_return_conditional_losses_93523ด
ซฒง
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsช 
annotationsช *
 z๚trace_0z๛trace_1
"
_generic_user_object
 "
trackable_dict_wrapper
0
ุ0
ู1"
trackable_list_wrapper
0
ุ0
ู1"
trackable_list_wrapper
 "
trackable_list_wrapper
ธ
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
layer_metrics
า	variables
ำtrainable_variables
ิregularization_losses
ึ__call__
+ื&call_and_return_all_conditional_losses
'ื"call_and_return_conditional_losses"
_generic_user_object
ํ
trace_02ฮ
'__inference_dense_8_layer_call_fn_93532ข
ฒ
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsช *
 ztrace_0

trace_02้
B__inference_dense_8_layer_call_and_return_conditional_losses_93543ข
ฒ
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsช *
 ztrace_0
!:	2dense_8/kernel
:2dense_8/bias
 "
trackable_dict_wrapper
j
90
:1
V2
W3
s4
t5
6
7
8
?9"
trackable_list_wrapper
ถ
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
19"
trackable_list_wrapper
0
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
B
,__inference_sequential_2_layer_call_fn_91779rescaling_2_input"ภ
ทฒณ
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsช 
annotationsช *
 
?B๛
,__inference_sequential_2_layer_call_fn_92592inputs"ภ
ทฒณ
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsช 
annotationsช *
 
?B๛
,__inference_sequential_2_layer_call_fn_92669inputs"ภ
ทฒณ
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsช 
annotationsช *
 
B
,__inference_sequential_2_layer_call_fn_92244rescaling_2_input"ภ
ทฒณ
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsช 
annotationsช *
 
B
G__inference_sequential_2_layer_call_and_return_conditional_losses_92810inputs"ภ
ทฒณ
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsช 
annotationsช *
 
B
G__inference_sequential_2_layer_call_and_return_conditional_losses_92965inputs"ภ
ทฒณ
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsช 
annotationsช *
 
คBก
G__inference_sequential_2_layer_call_and_return_conditional_losses_92340rescaling_2_input"ภ
ทฒณ
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsช 
annotationsช *
 
คBก
G__inference_sequential_2_layer_call_and_return_conditional_losses_92436rescaling_2_input"ภ
ทฒณ
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsช 
annotationsช *
 
ิBั
#__inference_signature_wrapper_92515rescaling_2_input"
ฒ
FullArgSpec
args 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsช *
 
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
฿B?
+__inference_rescaling_2_layer_call_fn_92970inputs"ข
ฒ
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsช *
 
๚B๗
F__inference_rescaling_2_layer_call_and_return_conditional_losses_92978inputs"ข
ฒ
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsช *
 
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
?Bฺ
)__inference_conv2d_10_layer_call_fn_92987inputs"ข
ฒ
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsช *
 
๘B๕
D__inference_conv2d_10_layer_call_and_return_conditional_losses_92998inputs"ข
ฒ
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsช *
 
.
90
:1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
?B๙
6__inference_batch_normalization_10_layer_call_fn_93011inputs"ด
ซฒง
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsช 
annotationsช *
 
?B๙
6__inference_batch_normalization_10_layer_call_fn_93024inputs"ด
ซฒง
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsช 
annotationsช *
 
B
Q__inference_batch_normalization_10_layer_call_and_return_conditional_losses_93042inputs"ด
ซฒง
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsช 
annotationsช *
 
B
Q__inference_batch_normalization_10_layer_call_and_return_conditional_losses_93060inputs"ด
ซฒง
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsช 
annotationsช *
 
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
ใBเ
/__inference_max_pooling2d_6_layer_call_fn_93065inputs"ข
ฒ
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsช *
 
?B๛
J__inference_max_pooling2d_6_layer_call_and_return_conditional_losses_93070inputs"ข
ฒ
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsช *
 
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
?Bฺ
)__inference_conv2d_11_layer_call_fn_93079inputs"ข
ฒ
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsช *
 
๘B๕
D__inference_conv2d_11_layer_call_and_return_conditional_losses_93090inputs"ข
ฒ
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsช *
 
.
V0
W1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
?B๙
6__inference_batch_normalization_11_layer_call_fn_93103inputs"ด
ซฒง
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsช 
annotationsช *
 
?B๙
6__inference_batch_normalization_11_layer_call_fn_93116inputs"ด
ซฒง
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsช 
annotationsช *
 
B
Q__inference_batch_normalization_11_layer_call_and_return_conditional_losses_93134inputs"ด
ซฒง
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsช 
annotationsช *
 
B
Q__inference_batch_normalization_11_layer_call_and_return_conditional_losses_93152inputs"ด
ซฒง
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsช 
annotationsช *
 
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
ใBเ
/__inference_max_pooling2d_7_layer_call_fn_93157inputs"ข
ฒ
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsช *
 
?B๛
J__inference_max_pooling2d_7_layer_call_and_return_conditional_losses_93162inputs"ข
ฒ
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsช *
 
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
?Bฺ
)__inference_conv2d_12_layer_call_fn_93171inputs"ข
ฒ
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsช *
 
๘B๕
D__inference_conv2d_12_layer_call_and_return_conditional_losses_93182inputs"ข
ฒ
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsช *
 
.
s0
t1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
?B๙
6__inference_batch_normalization_12_layer_call_fn_93195inputs"ด
ซฒง
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsช 
annotationsช *
 
?B๙
6__inference_batch_normalization_12_layer_call_fn_93208inputs"ด
ซฒง
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsช 
annotationsช *
 
B
Q__inference_batch_normalization_12_layer_call_and_return_conditional_losses_93226inputs"ด
ซฒง
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsช 
annotationsช *
 
B
Q__inference_batch_normalization_12_layer_call_and_return_conditional_losses_93244inputs"ด
ซฒง
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsช 
annotationsช *
 
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
?Bฺ
)__inference_conv2d_13_layer_call_fn_93253inputs"ข
ฒ
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsช *
 
๘B๕
D__inference_conv2d_13_layer_call_and_return_conditional_losses_93264inputs"ข
ฒ
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsช *
 
0
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
?B๙
6__inference_batch_normalization_13_layer_call_fn_93277inputs"ด
ซฒง
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsช 
annotationsช *
 
?B๙
6__inference_batch_normalization_13_layer_call_fn_93290inputs"ด
ซฒง
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsช 
annotationsช *
 
B
Q__inference_batch_normalization_13_layer_call_and_return_conditional_losses_93308inputs"ด
ซฒง
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsช 
annotationsช *
 
B
Q__inference_batch_normalization_13_layer_call_and_return_conditional_losses_93326inputs"ด
ซฒง
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsช 
annotationsช *
 
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
?Bฺ
)__inference_conv2d_14_layer_call_fn_93335inputs"ข
ฒ
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsช *
 
๘B๕
D__inference_conv2d_14_layer_call_and_return_conditional_losses_93346inputs"ข
ฒ
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsช *
 
0
0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
?B๙
6__inference_batch_normalization_14_layer_call_fn_93359inputs"ด
ซฒง
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsช 
annotationsช *
 
?B๙
6__inference_batch_normalization_14_layer_call_fn_93372inputs"ด
ซฒง
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsช 
annotationsช *
 
B
Q__inference_batch_normalization_14_layer_call_and_return_conditional_losses_93390inputs"ด
ซฒง
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsช 
annotationsช *
 
B
Q__inference_batch_normalization_14_layer_call_and_return_conditional_losses_93408inputs"ด
ซฒง
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsช 
annotationsช *
 
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
ใBเ
/__inference_max_pooling2d_8_layer_call_fn_93413inputs"ข
ฒ
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsช *
 
?B๛
J__inference_max_pooling2d_8_layer_call_and_return_conditional_losses_93418inputs"ข
ฒ
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsช *
 
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
?Bฺ
)__inference_flatten_2_layer_call_fn_93423inputs"ข
ฒ
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsช *
 
๘B๕
D__inference_flatten_2_layer_call_and_return_conditional_losses_93429inputs"ข
ฒ
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsช *
 
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
?Bุ
'__inference_dense_6_layer_call_fn_93438inputs"ข
ฒ
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsช *
 
๖B๓
B__inference_dense_6_layer_call_and_return_conditional_losses_93449inputs"ข
ฒ
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsช *
 
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
๏B์
)__inference_dropout_4_layer_call_fn_93454inputs"ด
ซฒง
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsช 
annotationsช *
 
๏B์
)__inference_dropout_4_layer_call_fn_93459inputs"ด
ซฒง
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsช 
annotationsช *
 
B
D__inference_dropout_4_layer_call_and_return_conditional_losses_93464inputs"ด
ซฒง
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsช 
annotationsช *
 
B
D__inference_dropout_4_layer_call_and_return_conditional_losses_93476inputs"ด
ซฒง
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsช 
annotationsช *
 
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
?Bุ
'__inference_dense_7_layer_call_fn_93485inputs"ข
ฒ
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsช *
 
๖B๓
B__inference_dense_7_layer_call_and_return_conditional_losses_93496inputs"ข
ฒ
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsช *
 
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
๏B์
)__inference_dropout_5_layer_call_fn_93501inputs"ด
ซฒง
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsช 
annotationsช *
 
๏B์
)__inference_dropout_5_layer_call_fn_93506inputs"ด
ซฒง
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsช 
annotationsช *
 
B
D__inference_dropout_5_layer_call_and_return_conditional_losses_93511inputs"ด
ซฒง
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsช 
annotationsช *
 
B
D__inference_dropout_5_layer_call_and_return_conditional_losses_93523inputs"ด
ซฒง
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsช 
annotationsช *
 
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
?Bุ
'__inference_dense_8_layer_call_fn_93532inputs"ข
ฒ
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsช *
 
๖B๓
B__inference_dense_8_layer_call_and_return_conditional_losses_93543inputs"ข
ฒ
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsช *
 
R
	variables
	keras_api

total

count"
_tf_keras_metric
c
	variables
	keras_api

total

count

_fn_kwargs"
_tf_keras_metric
0
0
1"
trackable_list_wrapper
.
	variables"
_generic_user_object
:  (2total
:  (2count
0
0
1"
trackable_list_wrapper
.
	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapperิ
 __inference__wrapped_model_91124ฏ4,-789:IJTUVWfgqrst|}?ถทวศุูDขA
:ข7
52
rescaling_2_input?????????เ
ช "1ช.
,
dense_8!
dense_8?????????์
Q__inference_batch_normalization_10_layer_call_and_return_conditional_losses_93042789:MขJ
Cข@
:7
inputs+???????????????????????????`
p 
ช "?ข<
52
0+???????????????????????????`
 ์
Q__inference_batch_normalization_10_layer_call_and_return_conditional_losses_93060789:MขJ
Cข@
:7
inputs+???????????????????????????`
p
ช "?ข<
52
0+???????????????????????????`
 ฤ
6__inference_batch_normalization_10_layer_call_fn_93011789:MขJ
Cข@
:7
inputs+???????????????????????????`
p 
ช "2/+???????????????????????????`ฤ
6__inference_batch_normalization_10_layer_call_fn_93024789:MขJ
Cข@
:7
inputs+???????????????????????????`
p
ช "2/+???????????????????????????`๎
Q__inference_batch_normalization_11_layer_call_and_return_conditional_losses_93134TUVWNขK
DขA
;8
inputs,???????????????????????????
p 
ช "@ข=
63
0,???????????????????????????
 ๎
Q__inference_batch_normalization_11_layer_call_and_return_conditional_losses_93152TUVWNขK
DขA
;8
inputs,???????????????????????????
p
ช "@ข=
63
0,???????????????????????????
 ฦ
6__inference_batch_normalization_11_layer_call_fn_93103TUVWNขK
DขA
;8
inputs,???????????????????????????
p 
ช "30,???????????????????????????ฦ
6__inference_batch_normalization_11_layer_call_fn_93116TUVWNขK
DขA
;8
inputs,???????????????????????????
p
ช "30,???????????????????????????๎
Q__inference_batch_normalization_12_layer_call_and_return_conditional_losses_93226qrstNขK
DขA
;8
inputs,???????????????????????????
p 
ช "@ข=
63
0,???????????????????????????
 ๎
Q__inference_batch_normalization_12_layer_call_and_return_conditional_losses_93244qrstNขK
DขA
;8
inputs,???????????????????????????
p
ช "@ข=
63
0,???????????????????????????
 ฦ
6__inference_batch_normalization_12_layer_call_fn_93195qrstNขK
DขA
;8
inputs,???????????????????????????
p 
ช "30,???????????????????????????ฦ
6__inference_batch_normalization_12_layer_call_fn_93208qrstNขK
DขA
;8
inputs,???????????????????????????
p
ช "30,???????????????????????????๒
Q__inference_batch_normalization_13_layer_call_and_return_conditional_losses_93308NขK
DขA
;8
inputs,???????????????????????????
p 
ช "@ข=
63
0,???????????????????????????
 ๒
Q__inference_batch_normalization_13_layer_call_and_return_conditional_losses_93326NขK
DขA
;8
inputs,???????????????????????????
p
ช "@ข=
63
0,???????????????????????????
 ส
6__inference_batch_normalization_13_layer_call_fn_93277NขK
DขA
;8
inputs,???????????????????????????
p 
ช "30,???????????????????????????ส
6__inference_batch_normalization_13_layer_call_fn_93290NขK
DขA
;8
inputs,???????????????????????????
p
ช "30,???????????????????????????๒
Q__inference_batch_normalization_14_layer_call_and_return_conditional_losses_93390?NขK
DขA
;8
inputs,???????????????????????????
p 
ช "@ข=
63
0,???????????????????????????
 ๒
Q__inference_batch_normalization_14_layer_call_and_return_conditional_losses_93408?NขK
DขA
;8
inputs,???????????????????????????
p
ช "@ข=
63
0,???????????????????????????
 ส
6__inference_batch_normalization_14_layer_call_fn_93359?NขK
DขA
;8
inputs,???????????????????????????
p 
ช "30,???????????????????????????ส
6__inference_batch_normalization_14_layer_call_fn_93372?NขK
DขA
;8
inputs,???????????????????????????
p
ช "30,???????????????????????????ท
D__inference_conv2d_10_layer_call_and_return_conditional_losses_92998o,-9ข6
/ข,
*'
inputs?????????เ
ช ".ข+
$!
0?????????v`
 
)__inference_conv2d_10_layer_call_fn_92987b,-9ข6
/ข,
*'
inputs?????????เ
ช "!?????????v`ต
D__inference_conv2d_11_layer_call_and_return_conditional_losses_93090mIJ7ข4
-ข*
(%
inputs?????????:N`
ช ".ข+
$!
0?????????
 
)__inference_conv2d_11_layer_call_fn_93079`IJ7ข4
-ข*
(%
inputs?????????:N`
ช "!?????????ถ
D__inference_conv2d_12_layer_call_and_return_conditional_losses_93182nfg8ข5
.ข+
)&
inputs?????????	
ช ".ข+
$!
0?????????
 
)__inference_conv2d_12_layer_call_fn_93171afg8ข5
.ข+
)&
inputs?????????	
ช "!?????????ถ
D__inference_conv2d_13_layer_call_and_return_conditional_losses_93264n|}8ข5
.ข+
)&
inputs?????????
ช ".ข+
$!
0?????????
 
)__inference_conv2d_13_layer_call_fn_93253a|}8ข5
.ข+
)&
inputs?????????
ช "!?????????ธ
D__inference_conv2d_14_layer_call_and_return_conditional_losses_93346p8ข5
.ข+
)&
inputs?????????
ช ".ข+
$!
0?????????
 
)__inference_conv2d_14_layer_call_fn_93335c8ข5
.ข+
)&
inputs?????????
ช "!?????????ฆ
B__inference_dense_6_layer_call_and_return_conditional_losses_93449`ถท0ข-
&ข#
!
inputs?????????
ช "&ข#

0?????????
 ~
'__inference_dense_6_layer_call_fn_93438Sถท0ข-
&ข#
!
inputs?????????
ช "?????????ฆ
B__inference_dense_7_layer_call_and_return_conditional_losses_93496`วศ0ข-
&ข#
!
inputs?????????
ช "&ข#

0?????????
 ~
'__inference_dense_7_layer_call_fn_93485Sวศ0ข-
&ข#
!
inputs?????????
ช "?????????ฅ
B__inference_dense_8_layer_call_and_return_conditional_losses_93543_ุู0ข-
&ข#
!
inputs?????????
ช "%ข"

0?????????
 }
'__inference_dense_8_layer_call_fn_93532Rุู0ข-
&ข#
!
inputs?????????
ช "?????????ฆ
D__inference_dropout_4_layer_call_and_return_conditional_losses_93464^4ข1
*ข'
!
inputs?????????
p 
ช "&ข#

0?????????
 ฆ
D__inference_dropout_4_layer_call_and_return_conditional_losses_93476^4ข1
*ข'
!
inputs?????????
p
ช "&ข#

0?????????
 ~
)__inference_dropout_4_layer_call_fn_93454Q4ข1
*ข'
!
inputs?????????
p 
ช "?????????~
)__inference_dropout_4_layer_call_fn_93459Q4ข1
*ข'
!
inputs?????????
p
ช "?????????ฆ
D__inference_dropout_5_layer_call_and_return_conditional_losses_93511^4ข1
*ข'
!
inputs?????????
p 
ช "&ข#

0?????????
 ฆ
D__inference_dropout_5_layer_call_and_return_conditional_losses_93523^4ข1
*ข'
!
inputs?????????
p
ช "&ข#

0?????????
 ~
)__inference_dropout_5_layer_call_fn_93501Q4ข1
*ข'
!
inputs?????????
p 
ช "?????????~
)__inference_dropout_5_layer_call_fn_93506Q4ข1
*ข'
!
inputs?????????
p
ช "?????????ช
D__inference_flatten_2_layer_call_and_return_conditional_losses_93429b8ข5
.ข+
)&
inputs?????????
ช "&ข#

0?????????
 
)__inference_flatten_2_layer_call_fn_93423U8ข5
.ข+
)&
inputs?????????
ช "?????????ํ
J__inference_max_pooling2d_6_layer_call_and_return_conditional_losses_93070RขO
HขE
C@
inputs4????????????????????????????????????
ช "HขE
>;
04????????????????????????????????????
 ล
/__inference_max_pooling2d_6_layer_call_fn_93065RขO
HขE
C@
inputs4????????????????????????????????????
ช ";84????????????????????????????????????ํ
J__inference_max_pooling2d_7_layer_call_and_return_conditional_losses_93162RขO
HขE
C@
inputs4????????????????????????????????????
ช "HขE
>;
04????????????????????????????????????
 ล
/__inference_max_pooling2d_7_layer_call_fn_93157RขO
HขE
C@
inputs4????????????????????????????????????
ช ";84????????????????????????????????????ํ
J__inference_max_pooling2d_8_layer_call_and_return_conditional_losses_93418RขO
HขE
C@
inputs4????????????????????????????????????
ช "HขE
>;
04????????????????????????????????????
 ล
/__inference_max_pooling2d_8_layer_call_fn_93413RขO
HขE
C@
inputs4????????????????????????????????????
ช ";84????????????????????????????????????ถ
F__inference_rescaling_2_layer_call_and_return_conditional_losses_92978l9ข6
/ข,
*'
inputs?????????เ
ช "/ข,
%"
0?????????เ
 
+__inference_rescaling_2_layer_call_fn_92970_9ข6
/ข,
*'
inputs?????????เ
ช ""?????????เ๗
G__inference_sequential_2_layer_call_and_return_conditional_losses_92340ซ4,-789:IJTUVWfgqrst|}?ถทวศุูLขI
Bข?
52
rescaling_2_input?????????เ
p 

 
ช "%ข"

0?????????
 ๗
G__inference_sequential_2_layer_call_and_return_conditional_losses_92436ซ4,-789:IJTUVWfgqrst|}?ถทวศุูLขI
Bข?
52
rescaling_2_input?????????เ
p

 
ช "%ข"

0?????????
 ์
G__inference_sequential_2_layer_call_and_return_conditional_losses_92810?4,-789:IJTUVWfgqrst|}?ถทวศุูAข>
7ข4
*'
inputs?????????เ
p 

 
ช "%ข"

0?????????
 ์
G__inference_sequential_2_layer_call_and_return_conditional_losses_92965?4,-789:IJTUVWfgqrst|}?ถทวศุูAข>
7ข4
*'
inputs?????????เ
p

 
ช "%ข"

0?????????
 ฯ
,__inference_sequential_2_layer_call_fn_917794,-789:IJTUVWfgqrst|}?ถทวศุูLขI
Bข?
52
rescaling_2_input?????????เ
p 

 
ช "?????????ฯ
,__inference_sequential_2_layer_call_fn_922444,-789:IJTUVWfgqrst|}?ถทวศุูLขI
Bข?
52
rescaling_2_input?????????เ
p

 
ช "?????????ฤ
,__inference_sequential_2_layer_call_fn_925924,-789:IJTUVWfgqrst|}?ถทวศุูAข>
7ข4
*'
inputs?????????เ
p 

 
ช "?????????ฤ
,__inference_sequential_2_layer_call_fn_926694,-789:IJTUVWfgqrst|}?ถทวศุูAข>
7ข4
*'
inputs?????????เ
p

 
ช "?????????์
#__inference_signature_wrapper_92515ฤ4,-789:IJTUVWfgqrst|}?ถทวศุูYขV
ข 
OชL
J
rescaling_2_input52
rescaling_2_input?????????เ"1ช.
,
dense_8!
dense_8?????????