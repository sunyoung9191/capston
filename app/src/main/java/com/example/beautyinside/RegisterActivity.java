package com.example.beautyinside;

import android.content.Intent;
import android.os.Bundle;
import android.text.TextUtils;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.EditText;
import android.widget.Toast;

import androidx.annotation.NonNull;
import androidx.appcompat.app.AppCompatActivity;

import com.google.android.gms.tasks.OnCompleteListener;
import com.google.android.gms.tasks.Task;
import com.google.firebase.auth.AuthResult;
import com.google.firebase.auth.FirebaseAuth;
import com.google.firebase.database.DatabaseReference;
import com.google.firebase.database.FirebaseDatabase;

public class RegisterActivity extends AppCompatActivity {

    private static final String TAG = "RegisterActivity";

    private FirebaseAuth mFirebaseAuth;
    private DatabaseReference mDatabaseRef;

    private EditText et_email, et_pwd;
    private Button btn_register;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_register);

        mFirebaseAuth = FirebaseAuth.getInstance();
        mDatabaseRef = FirebaseDatabase.getInstance().getReference("beautyinside");

        et_email = findViewById(R.id.et_email);
        et_pwd = findViewById(R.id.et_pwd);
        btn_register = findViewById(R.id.btn_register);


        btn_register.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                String strEmail = et_email.getText().toString();
                String strPwd = et_pwd.getText().toString();

                if (TextUtils.isEmpty(strEmail) || TextUtils.isEmpty(strPwd)) {
                    Toast.makeText(RegisterActivity.this, "이메일과 비밀번호를 모두 입력해주세요", Toast.LENGTH_SHORT).show();
                    return;
                }

                mFirebaseAuth.createUserWithEmailAndPassword(strEmail, strPwd)
                        .addOnCompleteListener(RegisterActivity.this, new OnCompleteListener<AuthResult>() {
                            @Override
                            public void onComplete(@NonNull Task<AuthResult> task) {
                                if (task.isSuccessful()) {
                                    // 계정 생성 성공 시
                                    FirebaseAuth auth = mFirebaseAuth;
                                    String uid = auth.getCurrentUser().getUid();

                                    UserAccount account = new UserAccount();
                                    account.setIdToken(uid);
                                    account.setEmailId(strEmail);
                                    account.setPassword(strPwd);

                                    mDatabaseRef.child("UserAccount").child(uid).setValue(account);

                                    Toast.makeText(RegisterActivity.this, "회원가입에 성공했습니다!", Toast.LENGTH_SHORT).show();
                                    startActivity(new Intent(RegisterActivity.this, LoginActivity.class));
                                    finish();
                                } else {
                                    Toast.makeText(RegisterActivity.this, "회원가입 실패: " + task.getException().getMessage(), Toast.LENGTH_SHORT).show();
                                    Log.w(TAG, "createUserWithEmail:failure", task.getException());
                                }
                            }
                        });
            }
        });
    }
}
