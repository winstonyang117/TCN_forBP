import torch
import timeit

def train_loop(trainLoader, model, device, Lr, metric_fn, loss_fn, is_pretrained = 0):
    start = timeit.default_timer()
    num_batches = len(trainLoader)
    if is_pretrained == 0:
        optimizer = torch.optim.Adam(model.parameters(), lr=Lr)
    else:
        for weights in model.encoder.parameters():
            weights.requires_grad = False
        params = filter(lambda p: p.requires_grad, model.parameters())
        optimizer = torch.optim.Adam(params, lr=1e-3)
    model.train()
    train_loss = 0
    train_mae = 0
    size = len(trainLoader.dataset)
    for batch, (data_batch, labels) in enumerate(trainLoader):
        data_batch = data_batch.to(device, dtype=torch.float)
        labels = labels.to(device)
        pred = model(data_batch)[:,0]
        # print(pred.shape, labels.shape)
        loss = loss_fn(pred, labels)
        metric = metric_fn(pred, labels)
        train_mae += metric
        optimizer.zero_grad() # if don't call zero_grad, the grad of each batch will be accumulated
        loss.backward()
        optimizer.step()
        # scheduler.step(epoch + batch / size)


        # print results according to each batch
        if batch % 1 == 0:
            # print('learning rate: ', scheduler.get_lr())
            loss, current = loss.item(), batch * len(data_batch)
            metric = metric.item()
            #print(f"loss: {loss:>7f}------ metric: {metric:>7f}  [{current:>5d}/{size:>5d}]")


    train_mae = train_mae / num_batches
    print(f"Train MAE: {train_mae:>7f}")
    stop = timeit.default_timer()
    print('It takes: '+str(stop - start)+ 's for training') 
 
def test_loop(dataloader, model, epoch, loss_fn, metric_fn, device):
    num_batches = len(dataloader)
    test_metric= 0
    loss = 0
    cnt = 0
    model.eval()
     
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)[:,0]
            loss += loss_fn(pred, y)
            test_metric += metric_fn(pred, y).item()
        loss /= num_batches
        test_metric /= num_batches
    print( f"Test MAE: {test_metric:>8f} \n")
    checkpoint = {
          "model": model.state_dict(),
          "epoch": epoch
      }
    if epoch % 5 == 0 and epoch != 0:
        torch.save(checkpoint, './train_model/ckpt_best_%s_systolic_%.2f.pth' %(str(epoch + 1), test_metric))
    
    return test_metric


