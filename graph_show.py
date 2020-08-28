from functions.plot import PlotGenerator
import os

graph_dir = r'./save/branch_1/history'
epoch_loss = 'ResNet101-DeepLabV3_epoch.txt'
valid_loss = 'ResNet101-DeepLabV3_valid.txt'
graph_store_dir = os.path.join(graph_dir, 'graphs')

train_plot = PlotGenerator(1, 'train loss', (20, 15), xlabel='epochs', ylabel='BCELoss')
train_plot.add_data(os.path.join(graph_dir, epoch_loss))
train_plot.add_set(name='Train loss', color='r')
train_plot.plot()
train_plot.save(os.path.join(graph_store_dir, 'train_loss.jpg'))
valid_plot = PlotGenerator(2, 'validation loss', (20, 15), xlabel='epochs', ylabel='BCELoss')
valid_plot.add_data(os.path.join(graph_dir, valid_loss))
valid_plot.add_set(name='Valid loss', color='y')
valid_plot.plot()
valid_plot.save(os.path.join(graph_store_dir, 'valid_loss.jpg'))
overlay_plot = PlotGenerator(3, 'Training history', (20,15), xlabel='epochs', ylabel='BCELoss')
overlay_plot.add_data(train_plot.data(0))
overlay_plot.add_data(valid_plot.data(0))
overlay_plot.add_set(data=train_plot.set(0))
overlay_plot.add_set(data=valid_plot.set(0))
overlay_plot.plot()
overlay_plot.save(os.path.join(graph_store_dir, 'Training History.jpg'))