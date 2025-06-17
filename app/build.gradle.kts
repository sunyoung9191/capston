plugins {
    alias(libs.plugins.android.application)
    alias(libs.plugins.kotlin.android)
    alias(libs.plugins.kotlin.kapt)
}
apply(plugin = "com.google.gms.google-services")
android {
    namespace = "com.example.beautyinside"
    compileSdk = 34

    defaultConfig {
        applicationId = "com.example.beautyinside"
        minSdk = 24
        targetSdk = 34
        versionCode = 1
        versionName = "1.0"
    }
    buildFeatures {
        viewBinding = true
    }
    buildTypes {
        release {
            isMinifyEnabled = false
        }
    }
    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_1_8
        targetCompatibility = JavaVersion.VERSION_1_8
    }

    kotlinOptions {
        jvmTarget = "1.8"
    }
}

dependencies {
    // Firebase 로그인 (이메일/비밀번호 로그인 기능용)
    implementation("com.google.firebase:firebase-auth:22.3.0")

    // Firebase Realtime Database (사용자 정보 저장용)
    implementation("com.google.firebase:firebase-database:20.3.0")

    // 기존 라이브러리들
    implementation(libs.appcompat)
    implementation(libs.material)
    implementation(libs.constraintlayout)
    implementation(libs.navigation.fragment.ktx)
    implementation(libs.navigation.ui.ktx)
    implementation("com.opencsv:opencsv:5.8")
    implementation("com.github.bumptech.glide:glide:4.16.0")
    kapt("com.github.bumptech.glide:compiler:4.16.0")
    implementation("com.squareup.retrofit2:retrofit:2.9.0")
    implementation("com.squareup.retrofit2:converter-gson:2.9.0")
    implementation("com.squareup.okhttp3:logging-interceptor:4.9.1")



}
