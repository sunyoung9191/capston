package com.example.beautyinside;

import android.content.Context;
import android.content.Intent;
import android.net.Uri;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.TextView;
import androidx.annotation.NonNull;
import androidx.recyclerview.widget.RecyclerView;
import com.bumptech.glide.Glide;
import com.example.beautyinside.hospital.AmondHospitalDetailActivity;
import com.example.beautyinside.hospital.BabaHospitalDetailActivity;
import com.example.beautyinside.hospital.ClasyHospitalDetailActivity;
import com.example.beautyinside.hospital.DmHospitalDetailActivity;
import com.example.beautyinside.hospital.DressHospitalDetailActivity;
import com.example.beautyinside.hospital.EsHospitalDetailActivity;
import com.example.beautyinside.hospital.HitHospitalDetailActivity;
import com.example.beautyinside.hospital.MadeHospitalDetailActivity;
import com.example.beautyinside.hospital.MarbleHospitalDetailActivity;
import com.example.beautyinside.hospital.MindHospitalDetailActivity;
import com.example.beautyinside.hospital.MlHospitalDetailActivity;
import com.example.beautyinside.hospital.NanaHospitalDetailActivity;
import com.example.beautyinside.hospital.OnairHospitalDetailActivity;
import com.example.beautyinside.hospital.PopHospitalDetailActivity;
import com.example.beautyinside.hospital.RugoHospitalDetailActivity;
import com.example.beautyinside.hospital.ThankyouHospitalDetailActivity;
import com.example.beautyinside.hospital.TsHospitalDetailActivity;
import com.example.beautyinside.hospital.WonderHospitalDetailActivity;
import com.example.beautyinside.hospital.YellowHospitalDetailActivity;
import com.example.beautyinside.hospital.YounoHospitalDetailActivity;

import java.util.List;
import android.util.Log;
import android.widget.Toast;

import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import java.io.InputStream;
import java.io.IOException;

public class SimilarReviewAdapter extends RecyclerView.Adapter<SimilarReviewAdapter.ViewHolder> {

    private List<ReviewItem> reviewList;
    private Context context;

    public SimilarReviewAdapter(Context context, List<ReviewItem> reviewList) {
        this.context = context;
        this.reviewList = reviewList;

    }




    @NonNull
    @Override
    public ViewHolder onCreateViewHolder(@NonNull ViewGroup parent, int viewType) {
        View view = LayoutInflater.from(parent.getContext())
                .inflate(R.layout.item_similar_review, parent, false);
        return new ViewHolder(view);

    }

    @Override
    public void onBindViewHolder(@NonNull ViewHolder holder, int position) {
        ReviewItem item = reviewList.get(position);
        AssetManager assetManager = context.getAssets();



        try {
            // 이미지 파일명만 추출해서 로딩
            String beforeName = item.getBefore().replace("assets/images/", "");
            String afterName = item.getAfter().replace("assets/images/", "");

            InputStream beforeStream = assetManager.open("images/" + beforeName);
            Bitmap beforeBitmap = BitmapFactory.decodeStream(beforeStream);
            holder.imageBefore.setImageBitmap(beforeBitmap);

            InputStream afterStream = assetManager.open("images/" + afterName);
            Bitmap afterBitmap = BitmapFactory.decodeStream(afterStream);
            holder.imageAfter.setImageBitmap(afterBitmap);

        } catch (IOException e) {
            Log.e("예은디버그", "이미지 로드 실패: " + e.getMessage(), e);
            holder.imageBefore.setImageResource(R.drawable.sample_before);
            holder.imageAfter.setImageResource(R.drawable.sample_after);
        }

        // 병원, 의사, 시술 정보 표시
        holder.textClinic.setText(item.getHospital());
        holder.textDoctor.setText(item.getDoctor());
        holder.textProcedure.setText(item.getProcedure());

        // ✅ 유사도 표시 추가
        holder.textSimilarity.setText(String.format("유사도: %.4f", item.getSimilarity()));

        holder.detailButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                String hospitalName = item.getHospital();
                Class<?> targetActivity = getDetailActivityClass(hospitalName);
                Intent intent = new Intent(context, targetActivity);
                intent.putExtra("hospitalName", hospitalName);
                Log.d("SimilarReviewAdapter", "병원 이름: " + hospitalName);
                Log.d("SimilarReviewAdapter", "이동할 클래스: " + targetActivity);
                context.startActivity(intent);
            }
        });

    }




    @Override
    public int getItemCount() {
        return reviewList.size();
    }

    public static class ViewHolder extends RecyclerView.ViewHolder {
        ImageView imageBefore, imageAfter;
        TextView textClinic, textDoctor, textProcedure, textSimilarity;  // ✅ 유사도 텍스트 추가!
        Button detailButton;

        public ViewHolder(@NonNull View itemView) {
            super(itemView);
            imageBefore = itemView.findViewById(R.id.image_before);
            imageAfter = itemView.findViewById(R.id.image_after);
            textClinic = itemView.findViewById(R.id.text_clinic);
            textDoctor = itemView.findViewById(R.id.text_doctor);
            textProcedure = itemView.findViewById(R.id.text_procedure);
            detailButton = itemView.findViewById(R.id.button_detail);

            textSimilarity = itemView.findViewById(R.id.text_similarity);  // ✅ 유사도 뷰 연결!
        }
    }


    private Class<?> getDetailActivityClass(String hospitalName) {
        if (hospitalName == null) {
            return MainActivity.class; // null 체크 추가
        }

        // 병원이름에 따라 상세 페이지 Activity 클래스 연결
        switch (hospitalName.replaceAll("\\s", "")) {
            case "바바성형외과":
                return BabaHospitalDetailActivity.class;
            case "아몬드성형외과":
                return AmondHospitalDetailActivity.class;
            case "나나성형외과":
                return NanaHospitalDetailActivity.class;
            case "팝성형외과":
                return PopHospitalDetailActivity.class;
            case "티에스성형외과":
                return TsHospitalDetailActivity.class;
            case "루호성형외과":
                return RugoHospitalDetailActivity.class;
            case "마인드성형외과":
                return MindHospitalDetailActivity.class;
            case "드레스성형외과":
                return DressHospitalDetailActivity.class;
            case "디엠성형외과":
                return DmHospitalDetailActivity.class;
            case "옐로우성형외과":
                return YellowHospitalDetailActivity.class;
            case "히트성형외과":
                return HitHospitalDetailActivity.class;
            case "마블성형외과":
                return MarbleHospitalDetailActivity.class;
            case "온에어성형외과":
                return OnairHospitalDetailActivity.class;
            case "유노성형외과":
                return YounoHospitalDetailActivity.class;
            case "메이드영성형외과":
                return MadeHospitalDetailActivity.class;
            case "땡큐성형외과":
                return ThankyouHospitalDetailActivity.class;
            case "원더풀성형외과":
                return WonderHospitalDetailActivity.class;
            case "이에스성형외과":
                return EsHospitalDetailActivity.class;
            case "일미리성형외과":
                return MlHospitalDetailActivity.class;
            case "클래시성형외과":
                return ClasyHospitalDetailActivity.class;
            default:
                Log.w("HospitalAdapter", "알 수 없는 병원 이름: " + hospitalName);
                return MainActivity.class;
        }
    }

}
