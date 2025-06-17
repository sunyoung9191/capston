package com.example.beautyinside;

import android.content.Context;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.ImageView;
import android.widget.LinearLayout;
import android.widget.TextView;
import androidx.annotation.NonNull;
import androidx.recyclerview.widget.RecyclerView;
import com.example.beautyinside.R;
import com.example.beautyinside.DoctorData;
import java.util.List;

public class DoctorListAdapter extends RecyclerView.Adapter<DoctorListAdapter.DoctorViewHolder> {

    private final Context context;
    private final List<DoctorData> doctorList;

    public DoctorListAdapter(Context context, List<DoctorData> doctorList) {
        this.context = context;
        this.doctorList = doctorList;
    }

    @NonNull
    @Override
    public DoctorViewHolder onCreateViewHolder(@NonNull ViewGroup parent, int viewType) {
        View view = LayoutInflater.from(context).inflate(R.layout.item_doctor, parent, false);
        return new DoctorViewHolder(view);
    }

    @Override
    public void onBindViewHolder(@NonNull DoctorViewHolder holder, int position) {
        DoctorData doctor = doctorList.get(position);
        holder.textDoctorName.setText(doctor.getName());
        holder.imageDoctor.setImageResource(doctor.getImageResId());

        holder.tagContainer.removeAllViews();
        for (String tag : doctor.getTags()) {
            TextView tagView = new TextView(context);
            tagView.setText(tag);
            tagView.setTextSize(12);
            tagView.setTextColor(0xFF555555);
            tagView.setBackgroundResource(R.drawable.tag_background); // 배경 drawable 필요
            tagView.setPadding(16, 8, 16, 8);
            LinearLayout.LayoutParams params = new LinearLayout.LayoutParams(
                    ViewGroup.LayoutParams.WRAP_CONTENT,
                    ViewGroup.LayoutParams.WRAP_CONTENT);
            params.setMargins(0, 0, 16, 0);
            tagView.setLayoutParams(params);
            holder.tagContainer.addView(tagView);
        }
    }

    @Override
    public int getItemCount() {
        return doctorList.size();
    }

    public static class DoctorViewHolder extends RecyclerView.ViewHolder {
        ImageView imageDoctor;
        TextView textDoctorName;
        LinearLayout tagContainer;

        public DoctorViewHolder(@NonNull View itemView) {
            super(itemView);
            imageDoctor = itemView.findViewById(R.id.imageDoctor);
            textDoctorName = itemView.findViewById(R.id.textDoctorName);
            tagContainer = itemView.findViewById(R.id.tagContainer);
        }
    }
}
