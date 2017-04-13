# tf函数使用记录

**tf.concat**`(values, axis, name='concat')`

合并数据，axis为0合并行，列保留。为1行保留，列合并。

```python
t1 = [[1, 2, 3], [4, 5, 6]]
t2 = [[7, 8, 9], [10, 11, 12]]
tf.concat([t1, t2], 0) ==> [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]
tf.concat([t1, t2], 1) ==> [[1, 2, 3, 7, 8, 9], [4, 5, 6, 10, 11, 12]]

```

**tf.expand_dims**`(input, axis=None, name=None, dim=None)`

给张量增加一个维度。input为张量，axis为插入维度的位置。

```python
# 't' is a tensor of shape [2]
#插入维度的位置尺度为1.
shape(expand_dims(t, 0)) ==> [1, 2]
shape(expand_dims(t, 1)) ==> [2, 1]
shape(expand_dims(t, -1)) ==> [2, 1]

l=[1,2,1,3,4,2,1]
tf.expand_dims(l, 0)==>[[1,2,1,3,4,2,1]]
tf.expand_dims(l, 1)==>[[1],[2],[1],[3],[4],[2],[1]]

# 't2' is a tensor of shape [2, 3, 5]
shape(expand_dims(t2, 0)) ==> [1, 2, 3, 5]
shape(expand_dims(t2, 2)) ==> [2, 3, 1, 5]
shape(expand_dims(t2, 3)) ==> [2, 3, 5, 1]
```