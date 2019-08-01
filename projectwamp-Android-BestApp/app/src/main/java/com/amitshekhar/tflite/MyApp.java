package com.amitshekhar.tflite;

import android.app.Application;
import android.os.SystemClock;

public class MyApp extends Application {
    @Override
    public void onCreate(){
        super.onCreate();
        SystemClock.sleep(1000);
    }
}
