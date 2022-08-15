use anyhow::Context as _;
use image::codecs::png::PngEncoder;
use image::{DynamicImage, ImageEncoder};
use serde::{Deserialize, Serialize};
use serde_json::json;
use serde_json::Value;

const CLOUD_VISION_URI: &str = "https://vision.googleapis.com/v1/images:annotate";

pub struct ImageGCV {
    base64_data: String,
}

impl ImageGCV {
    pub fn from_image(image: &DynamicImage) -> anyhow::Result<Self> {
        let mut buf = vec![];
        {
            let encoder = PngEncoder::new(&mut buf);

            encoder.write_image(
                image.as_bytes(),
                image.width(),
                image.height(),
                image.color(),
            )?;
        }

        Ok(Self {
            base64_data: base64::encode(buf),
        })
    }
}

#[derive(Serialize, Deserialize, Debug)]
pub struct TextAnnotation {
    pub locale: Option<String>,
    pub description: String,
    #[serde(rename = "boundingPoly")]
    pub bounding_poly: Polygon,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct Polygon {
    pub vertices: Vec<Point>,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct Point {
    pub x: Option<i64>,
    pub y: Option<i64>,
}

/// Client for google cloud vision
pub struct Client {
    credential: String,
}

impl Client {
    pub fn new(apikey: &str) -> Self {
        Self {
            credential: apikey.to_string(),
        }
    }

    /// The most commonly used methods are
    /// ```bash
    /// export GOOGLE_APPLICATION_CREDENTIALS=/path/to/key.json
    /// export GCV_API_KEY=`gcloud auth application-default print-access-token`
    /// ```
    pub fn new_from_env() -> Option<Self> {
        Some(Self::new(std::env::var("GCV_API_KEY").ok()?.as_str()))
    }

    pub async fn text_annotations(&self, image: &ImageGCV) -> anyhow::Result<Vec<TextAnnotation>> {
        let request = json!({
           "requests" : [
               {
                    "image": {
                        "content": image.base64_data
                    },
                    "features": [
                        {
                            "type": "DOCUMENT_TEXT_DETECTION"
                        }
                    ],
               }
           ]
        });

        let response = reqwest::Client::new()
            .post(CLOUD_VISION_URI)
            .header("Authorization", format!("Bearer {}", self.credential))
            .json(&request)
            .send()
            .await?;

        let json_response: Value = response.json().await?;

        let err = &json_response["error"];

        if err.is_object() {
            return Err(anyhow::anyhow!(err.to_string()));
        }

        let text_annotations_value = &json_response["responses"][0]["textAnnotations"];

        Ok(text_annotations_value
            .as_array()
            .with_context(|| format!("text_annotations must be array: {}", json_response))?
            .iter()
            .map(|x| {
                serde_json::from_value(x.clone()).expect("textAnnotation json value parse error")
            })
            .collect())
    }
}

#[cfg(test)]
mod tests {
    use crate::{Client, ImageGCV, TextAnnotation};
    use image::io::Reader as ImageReader;
    use serde_json::Value;

    #[tokio::test]
    async fn it_works() {
        let client = Client::new(
            std::env::var("GCV_API_KEY")
                .expect("please set GCV_API_KEY")
                .as_str(),
        );
        let mut image = ImageReader::open("test/test.png")
            .unwrap()
            .decode()
            .unwrap();
        let gcv_image = ImageGCV::from_image(&image).unwrap();
        let result = client.text_annotations(&gcv_image).await;
        assert!(result.is_ok());
        assert_eq!(result.as_ref().unwrap().len(), 2);
        assert_eq!(result.as_ref().unwrap()[0].description, "44097050");
    }
    #[test]
    fn encode_() {
        let mut image = ImageReader::open("test/test10.png")
            .unwrap()
            .decode()
            .unwrap();
        eprintln!("{} {}", image.width(), image.height());
        let gcv_image = ImageGCV::from_image(&image).unwrap();

        assert_ne!(gcv_image.base64_data.len(), 0);
    }
    #[test]
    fn deserialize_text_annotation() {
        let json_text = r#"{
  "description": "ENGINE",
  "boundingPoly": {
    "vertices": [
      {
        "x": 1222,
        "y": 1771
      },
      {
        "x": 1944,
        "y": 1834
      },
      {
        "x": 1930,
        "y": 1992
      },
      {
        "x": 1208,
        "y": 1928
      }
    ]
  }
}"#;
        let json_value: Value = serde_json::from_str(json_text).unwrap();
        let text_annotation: TextAnnotation = serde_json::from_value(json_value).unwrap();

        assert_eq!(text_annotation.description, "ENGINE");
        assert_eq!(text_annotation.locale, None);
        assert_eq!(text_annotation.bounding_poly.vertices.len(), 4);
        assert_eq!(text_annotation.bounding_poly.vertices[0].x, Some(1222));
        assert_eq!(text_annotation.bounding_poly.vertices[0].y, Some(1771));
    }
}
