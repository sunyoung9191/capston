package com.example.beautyinside;

import android.content.Context;
import android.content.Intent;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.ImageView;
import android.widget.TextView;
import androidx.annotation.NonNull;
import androidx.recyclerview.widget.RecyclerView;
import com.bumptech.glide.Glide;
import java.util.List;
import android.util.Log;
import android.view.View;

public class ReviewAdapter extends RecyclerView.Adapter<ReviewAdapter.ViewHolder> {
    private List<ReviewData> reviewList;
    private Context context;

    public ReviewAdapter(Context context, List<ReviewData> reviewList) {
        this.context = context;
        this.reviewList = reviewList;
    }

    public static class ViewHolder extends RecyclerView.ViewHolder {
        TextView  textNickname;
        TextView textProcedure;
        TextView doctorText;
        ImageView beforeImageView;
        ImageView afterImageView;

        public ViewHolder(View itemView) {
            super(itemView);
            textNickname = itemView.findViewById(R.id.textNickname);
            textProcedure = itemView.findViewById(R.id.textProcedure);
            doctorText = itemView.findViewById(R.id.doctorText);
            beforeImageView = itemView.findViewById(R.id.imageBefore);
            afterImageView = itemView.findViewById(R.id.imageAfter);
        }
    }

    @NonNull
    @Override
    public ReviewAdapter.ViewHolder onCreateViewHolder(@NonNull ViewGroup parent, int viewType) {
        View view = LayoutInflater.from(parent.getContext())
                .inflate(R.layout.activity_review_list, parent, false);
        return new ViewHolder(view);
    }

    @Override
    public void onBindViewHolder(@NonNull ReviewAdapter.ViewHolder holder, int position) {
        ReviewData data = reviewList.get(position);
        holder.textNickname.setText(data.getUserName());
        String raw = data.getType();  // 예: "자연유착,윗트임"
        String[] procedures = data.getType().split(",");
        StringBuilder sb = new StringBuilder();
        for (String p : procedures) {
            sb.append("#").append(p.trim()).append(" ");
        }
        holder.textProcedure.setText(sb.toString().trim());

        holder.doctorText.setText(data.getDoctorName());


        String beforePath = "file:///android_asset/images/" + data.getBeforeImage();
        String afterPath = "file:///android_asset/images/" + data.getAfterImage();
        try {
            Glide.with(context)
                    .load(beforePath)
                    .error(R.drawable.logo)
                    .into(holder.beforeImageView);

            Glide.with(context)
                    .load(afterPath)
                    .into(holder.afterImageView);
        } catch (Exception e) {
            Log.e("ReviewDebug", "❌ Glide 이미지 로딩 실패: " + beforePath + ", " + afterPath, e);
        }
        Glide.with(context).load(beforePath).into(holder.beforeImageView);
        Glide.with(context).load(afterPath).into(holder.afterImageView);

        holder.itemView.setOnClickListener(v -> {
            Intent intent = new Intent(context, ReviewDetailActivity.class);
            intent.putExtra("reviewData", data); // Serializable
            context.startActivity(intent);
        });
    }

    @Override
    public int getItemCount() {
        return reviewList.size();
    }
}
