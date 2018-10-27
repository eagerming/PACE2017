import pickle

import dataset

train = None
with open('data/traindata_small.pkl', 'rb') as f:
    train = pickle.load(f)

user_input = train['user']
item_input = train['spot']
ui_label = train['label']
data = dataset.Dataset('_small')

try:
    with open('data/train_context_small.pkl', 'rb') as f:
        contexts = pickle.load(f)
except Exception:
    data.generateContextLabels()
    contexts = data.context_data
    try:
        with open('data/train_context_small.pkl', 'wb') as handle:
            pickle.dump(contexts, handle, protocol=pickle.HIGHEST_PROTOCOL)
    except:
        pass



u_context, s_context = contexts['user_context'], contexts['spot_context']
train_data = {}
train_data['user_input'] = user_input
train_data['item_input'] = item_input
train_data['ui_label'] = ui_label
train_data['u_context'] = u_context
train_data['s_context'] = s_context

# a = 1
#
# with open('data/Rating_gowalla.txt', 'w') as f:
#     for userID, itemID, rating in zip(user_input, item_input, ui_label):
#         f.write("%d %d %f\n" % (userID + 1, itemID + 1, rating))
#
# with open('data/user_network.txt','w') as f:
#     for userI, user_context in enumerate(u_context):
#         friends = [friend for friend in enumerate(user_context) if friend[1] > 0]
#         for friend in friends:
#             f.write("%d, %d, %f\n" % (userI + 1, friend[0] + 1, friend[1]))
#
#
# with open('data/item_network.txt','w') as f:
#     for userI, user_context in enumerate(s_context):
#         friends = [friend for friend in enumerate(user_context) if friend[1] > 0]
#         for friend in friends:
#             f.write("%d, %d, %f\n" % (userI + 1, friend[0] + 1, friend[1]))


