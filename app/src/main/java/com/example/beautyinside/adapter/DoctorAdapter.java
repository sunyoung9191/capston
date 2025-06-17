package com.example.beautyinside.adapter;

import android.content.Context;
import android.content.Intent;
import android.util.Log;

import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.ImageView;
import android.widget.TextView;

import androidx.annotation.NonNull;
import androidx.recyclerview.widget.RecyclerView;

import com.example.beautyinside.R;
import com.example.beautyinside.model.Hospital;
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

public class DoctorAdapter extends RecyclerView.Adapter<DoctorAdapter.ViewHolder> {

    private final Context context;
    private final List<Hospital> hospitalList;

    //  Context 포함한 생성자
    public DoctorAdapter(Context context, List<Hospital> hospitalList) {
        this.context = context;
        this.hospitalList = hospitalList;
    }

    @NonNull
    @Override
    public DoctorAdapter.ViewHolder onCreateViewHolder(@NonNull ViewGroup parent, int viewType) {
        View view = LayoutInflater.from(context).inflate(R.layout.item_hospital, parent, false);
        return new ViewHolder(view);
    }

    @Override
    public void onBindViewHolder(@NonNull DoctorAdapter.ViewHolder holder, int position) {
        Hospital hospital = hospitalList.get(position);

        holder.hospitalName.setText(hospital.getName());
        holder.hospitalDesc.setText(hospital.getDescription());

        //  대표 이미지 로딩 ("imageKey_1" 형식으로 동적 불러오기)
        String imageKey = hospital.getImageKey(); // ex: "pop"
        int imageResId = context.getResources().getIdentifier(
                imageKey + "_1", "drawable", context.getPackageName());
        holder.hospitalImage.setImageResource(imageResId);

        // 병원 클릭 시 상세 페이지 이동
        holder.itemView.setOnClickListener(v -> {

            if (hospital.getName().equals("팝 성형외과")) {

                Intent intent = new Intent(holder.itemView.getContext(), PopHospitalDetailActivity.class);
                holder.itemView.getContext().startActivity(intent);

            }

            if (hospital.getName().equals("티에스 성형외과")) {
                Intent intent = new Intent(holder.itemView.getContext(), TsHospitalDetailActivity.class);
                holder.itemView.getContext().startActivity(intent);

            }

            if (hospital.getName().equals("아몬드 성형외과")) {
                Intent intent = new Intent(holder.itemView.getContext(), AmondHospitalDetailActivity.class);
                holder.itemView.getContext().startActivity(intent);
            }

            if (hospital.getName().equals("바바 성형외과")) {
                Intent intent = new Intent(holder.itemView.getContext(), BabaHospitalDetailActivity.class);
                holder.itemView.getContext().startActivity(intent);
            }

            if (hospital.getName().equals("클래시 성형외과")) {
                Intent intent = new Intent(holder.itemView.getContext(), ClasyHospitalDetailActivity.class);
                holder.itemView.getContext().startActivity(intent);
            }

            if (hospital.getName().equals("드레스 성형외과")) {
                Intent intent = new Intent(holder.itemView.getContext(), DressHospitalDetailActivity.class);
                holder.itemView.getContext().startActivity(intent);
            }

            if (hospital.getName().equals("이에스 성형외과")) {
                Intent intent = new Intent(holder.itemView.getContext(), EsHospitalDetailActivity.class);
                holder.itemView.getContext().startActivity(intent);
            }

            if (hospital.getName().equals("히트 성형외과")) {
                Intent intent = new Intent(holder.itemView.getContext(), HitHospitalDetailActivity.class);
                holder.itemView.getContext().startActivity(intent);
            }

            if (hospital.getName().equals("메이드영 성형외과")) {
                Intent intent = new Intent(holder.itemView.getContext(), MadeHospitalDetailActivity.class);
                holder.itemView.getContext().startActivity(intent);
            }

            if (hospital.getName().equals("마블 성형외과")) {
                Intent intent = new Intent(holder.itemView.getContext(), MarbleHospitalDetailActivity.class);
                holder.itemView.getContext().startActivity(intent);
            }

            if (hospital.getName().equals("마인드 성형외과")) {
                Intent intent = new Intent(holder.itemView.getContext(), MindHospitalDetailActivity.class);
                holder.itemView.getContext().startActivity(intent);
            }

            if (hospital.getName().equals("일미리 성형외과")) {
                Intent intent = new Intent(holder.itemView.getContext(), MlHospitalDetailActivity.class);
                holder.itemView.getContext().startActivity(intent);
            }

            if (hospital.getName().equals("나나 성형외과")) {
                Intent intent = new Intent(holder.itemView.getContext(), NanaHospitalDetailActivity.class);
                holder.itemView.getContext().startActivity(intent);
            }
            if (hospital.getName().equals("온에어 성형외과")) {
                Intent intent = new Intent(holder.itemView.getContext(), OnairHospitalDetailActivity.class);
                holder.itemView.getContext().startActivity(intent);
            }
            if (hospital.getName().equals("루호 성형외과")) {
                Intent intent = new Intent(holder.itemView.getContext(), RugoHospitalDetailActivity.class);
                holder.itemView.getContext().startActivity(intent);
            }
            if (hospital.getName().equals("땡큐 성형외과")) {
                Intent intent = new Intent(holder.itemView.getContext(), ThankyouHospitalDetailActivity.class);
                holder.itemView.getContext().startActivity(intent);
            }
            if (hospital.getName().equals("원더풀 성형외과")) {
                Intent intent = new Intent(holder.itemView.getContext(), WonderHospitalDetailActivity.class);
                holder.itemView.getContext().startActivity(intent);
            }
            if (hospital.getName().equals("옐로우 성형외과")) {
                Intent intent = new Intent(holder.itemView.getContext(), YellowHospitalDetailActivity.class);
                holder.itemView.getContext().startActivity(intent);
            }
            if (hospital.getName().equals("유노 성형외과")) {
                Intent intent = new Intent(holder.itemView.getContext(), YounoHospitalDetailActivity.class);
                holder.itemView.getContext().startActivity(intent);
            }
            if (hospital.getName().equals("디엠 성형외과")) {
                Intent intent = new Intent(holder.itemView.getContext(), DmHospitalDetailActivity.class);
                holder.itemView.getContext().startActivity(intent);
            }
        });
    }

    @Override
    public int getItemCount() {
        return hospitalList.size();
    }

    public static class ViewHolder extends RecyclerView.ViewHolder {
        ImageView hospitalImage;
        TextView hospitalName;
        TextView hospitalDesc;

        public ViewHolder(@NonNull View itemView) {
            super(itemView);
            hospitalImage = itemView.findViewById(R.id.hospital_image);
            hospitalName = itemView.findViewById(R.id.hospital_name);
            hospitalDesc = itemView.findViewById(R.id.hospital_desc);
        }
    }
}
