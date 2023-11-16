import lmdb

env1 = lmdb.open('/media/space/T7/LMDB/got10k_train_resplit', readonly=True, lock=False, readahead=False, meminit=False)
handle1 = env1.begin(write=False)
database_1 = handle1.cursor()

env2 = lmdb.open('/media/space/T7/LMDB/got10k_val_resplit', readonly=True, lock=False, readahead=False, meminit=False)
handle2 = env2.begin(write=False)
database_2 = handle2.cursor()

env3 = lmdb.open('/home/space/Documents/got', map_size=1099511627776, readonly=False,
                 meminit=False, map_async=True)  # max(B) 1TB
txn_3 = env3.begin(write=True)

count = 0
# 遍历数据库
for (key, value) in database_1:
    # 将数据放到结果数据库事务中
    txn_3.put(key, value)
    count += 1
    if (count % 1000 == 0):
        # 将数据写入数据库，必须的，否则数据不会写入到数据库中
        txn_3.commit()
        count = 0
        txn_3 = env3.begin(write=True)
    print('lmdb1 1000 ok')

if (count % 1000 != 0):
    txn_3.commit()
    count = 0
    txn_3 = env3.begin(write=True)
    print('lmdb1 ok')

for (key, value) in database_2:
    txn_3.put(key, value)
    count += 1
    if (count % 1000 == 0):
        txn_3.commit()
        count = 0
        txn_3 = env3.begin(write=True)
    print('lmdb2 1000 ok')

if (count % 1000 != 0):
    txn_3.commit()
    count = 0
    txn_3 = env3.begin(write=True)
    print('lmdb2 ok')

# 关闭lmdb
env1.close()
env2.close()
env3.close()

print('Merge success!')

# 输出结果lmdb的状态信息，可以看到数据是否合并成功
print(env3.stat())
