import torch
from utils.visualize import save_loss


def fit_multimodal(
    username,
    model,
    train_loader,
    loss_fn,
    optimizer,
    val_loader,
    early_stopping,
    loss_logger,
    scheduler
):
    def _train_step(model, dataloader, loss_fn, optimizer, loss_logger):
        model.train()

        # data = [self.joint_inputs[idx], self.image_inputs[idx],\
            #  self.tactile_inputs[idx], self.targets[idx]]
    
        for data in dataloader:

            data = [d.to(model.device) for d in data]
            j, img, tac, tar = data
        
            inputs = [j, img, tac]
            targets = tar

            batch_size = inputs[0].shape[0]

            if model.is_rnn:
                model.init_hidden(batch_size)

            predictions, *_= model(inputs)

            loss = loss_fn(predictions, targets)
            
            loss_logger.train += loss

            optimizer.zero_grad()
            
            loss.backward()
            
            optimizer.step()      


        loss_logger.train.calc()

        return loss_logger
 

    def _validation_step(model, dataloader, loss_fn, loss_logger):
        model.eval()

        for data in dataloader:
            
            data = [d.to(model.device) for d in data]
            j, img, tac, tar = data
        
            inputs = [j, img, tac]
            targets = tar

            batch_size = inputs[0].shape[0]

            if model.is_rnn:
                model.init_hidden(batch_size)

            predictions, *_ = model(inputs)
            loss = loss_fn(predictions, targets)
            loss_logger.val += loss

        loss_logger.val.calc()

        return loss_logger

    for epoch in range(30000):
     
        _train_step(model, train_loader, loss_fn, optimizer, loss_logger)
      
        _validation_step(model, val_loader, loss_fn, loss_logger)
        
        print(f'epoch: {epoch+1}', end=' ')
        print(f'train_loss: {loss_logger.train.loss[-1]}', end=' ')
        print(f'val_loss: {loss_logger.val.loss[-1]}', end=' ')

        early_stopping(loss=loss_logger.val.loss[-1], model=model)
        if early_stopping.early_stop:
            break
        if scheduler:
            scheduler.step(loss_logger.val.loss[-1])
        if epoch % 100 == 0:
            save_loss(username, loss_logger)
            # f'{username}/outputs/learning_results/losses' にlossを保存
        if (epoch+1) % 100 ==0:
            torch.save(
                model.state_dict(),
                f'{username}/models/{early_stopping.model_name}/model_e{epoch+1}.pth')
            

def fit_enc_recon(
    username,
    model,
    train_loader,
    loss_fn,
    optimizer,
    val_loader,
    early_stopping,
    loss_logger,
    scheduler
):
    def _train_step(model, dataloader, loss_fn, optimizer, loss_logger):
        model.train()
    
        for data in dataloader:

             # data = [self.image_inputs[idx], self.targets[idx]]

            data = [d.to(model.device) for d in data]
            input, target = data

            batch_size = input.shape[0]

            if model.is_rnn:
                model.init_hidden(batch_size)

            predictions = model(input)

            loss = loss_fn(predictions, target)
            
            loss_logger.train += loss

            optimizer.zero_grad()
            
            loss.backward()
            
            optimizer.step()      

        loss_logger.train.calc()

        return loss_logger
 

    def _validation_step(model, dataloader, loss_fn, loss_logger):
        model.eval()

        for data in dataloader:
            
            data = [d.to(model.device) for d in data]
            input, target = data

            batch_size = input.shape[0]

            if model.is_rnn:
                model.init_hidden(batch_size)

            predictions = model(input)

            loss = loss_fn(predictions, target)

            loss_logger.val += loss

        loss_logger.val.calc()

        return loss_logger

    for epoch in range(30000):
     
        _train_step(model, train_loader, loss_fn, optimizer, loss_logger)
      
        _validation_step(model, val_loader, loss_fn, loss_logger)
        
        print(f'epoch: {epoch+1}', end=' ')
        print(f'train_loss: {loss_logger.train.loss[-1]}', end=' ')
        print(f'val_loss: {loss_logger.val.loss[-1]}', end=' ')

        early_stopping(loss=loss_logger.val.loss[-1], model=model)
        if early_stopping.early_stop:
            break
        if scheduler:
            scheduler.step(loss_logger.val.loss[-1])
        if epoch % 100 == 0:
            save_loss(username, loss_logger)
            # f'{username}/outputs/learning_results/losses' にlossを保存
        if (epoch+1) % 100 ==0:
            torch.save(
                model.state_dict(),
                f'{username}/models/{early_stopping.model_name}/model_e{epoch+1}.pth')
        
