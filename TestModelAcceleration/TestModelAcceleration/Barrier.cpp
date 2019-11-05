#include "Barrier.h"

Barrier::Barrier(std::size_t count) {
	mThreshold(count);
	mCount(count);
	mGeneration(0);
}

void Barrier::wait() {

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