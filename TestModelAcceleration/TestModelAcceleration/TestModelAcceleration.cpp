// TestModelAcceleration.cpp: 定义应用程序的入口点。
//

#include "TestModelAcceleration.h"
using namespace std;


class Barrier {

public:
	explicit Barrier(std::size_t count) {
		mThreshold = count;
		mCount = count;
		mGeneration = 0;
	}
	void wait() {
		std::unique_lock<std::mutex>lLock(mMutex);
		auto lGen = mGeneration;
		if ((--mCount) == 0) {
			mGeneration++;
			mCount = mThreshold;
			mCond.notify_all();
		}
		else {

			mCond.wait(lLock, [this, lGen] {return lGen != mGeneration; });
		}
	}

private:
	std::mutex mMutex;
	std::condition_variable mCond;
	std::size_t mCount;
	std::size_t mThreshold;
	std::size_t mGeneration;


};


int batch_size = 32;
int learning_rate = 0.01;
int num_epoch = 2;
mutex trainMutex;

Barrier cyclicBarrier(num_epoch + 1);
string str = "C:\\Users\\tchennech\\Documents\\ModelAcceleration\\TestModelAcceleration\\TestModelAcceleration\\test_script_model.pt";

torch::jit::script::Module model = torch::jit::load(str);



auto dataset = torch::data::datasets::MNIST("C:\\Users\\tchennech\\Documents\\ModelAcceleration\\TestModelAcceleration\\TestModelAcceleration\\mnist")
.map(torch::data::transforms::Normalize<>(0.13066062, 0.30810776))
.map(torch::data::transforms::Stack<>());

auto dataloader = torch::data::make_data_loader(move(dataset), torch::data::DataLoaderOptions().batch_size(32).workers(1));



void train_every_epoch(torch::optim::Adam optimizer){
	try {

	
		vector<torch::IValue> inputs;
		int idx = 0;
		cout << &optimizer << endl;
		for (torch::data::Example<>& batch : *dataloader) {
			idx++;
			torch::Tensor data = batch.data;
			torch::Tensor target = batch.target;
			data = data.to(at::kCUDA);
			target = target.to(at::kCUDA);
			inputs.push_back(data);
			torch::Tensor loss;
			if (trainMutex.try_lock()) {
				torch::Tensor output = model.forward(inputs).toTensor().to(at::kCUDA);
				loss = torch::nll_loss(output, target);
				optimizer.zero_grad();
				loss.backward();
				optimizer.step();
				trainMutex.unlock();
			}

			if (idx % 100 == 0) {

				cout << "loss: " << loss.item() << endl;
			}
			inputs.clear();
		}
		cyclicBarrier.wait();
	}
	catch (exception & e) {
		cerr << e.what() << endl;
	}
	
}

int main()
{
	
	try {
	
		
		cout << "load ok" << endl;

		if (torch::cuda::is_available()) {
			model.to(at::kCUDA);
		}
	}
	catch (exception &e) {
		cerr << e.what() << endl;
		cout << "load fail" << endl;
	}
	

	try {
	
		
		//auto dataset = torch::data::make_data_loader(std)

	

		vector<at::Tensor>parameters;

		for (const auto& parameter : model.get_parameters()) {
			parameters.push_back(parameter.value().toTensor().to(at::kCUDA));
		}

		torch::optim::Adam optimizer = torch::optim::Adam(parameters, torch::optim::AdamOptions(2e-4).learning_rate(learning_rate).beta1(0.5));
		
		double st, end;
		//parameters.push_back()
	
		//std::size_t epochs = 101;
		cout <<"first optimizer: " << &std::ref(optimizer) << endl;
		st = clock();
		for (int epoch = 0; epoch < num_epoch; epoch++) {
			thread th(train_every_epoch, std::ref(optimizer));
			th.detach();
			
		}
		cyclicBarrier.wait();
		end = clock();
		cout << "cost: " << end - st << endl;
	
	}
	catch (exception & e) {
		cerr << e.what() << endl;
	}


	return 0;
}
 