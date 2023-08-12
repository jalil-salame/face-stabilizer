use iced::Sandbox;
use log::error;

macro_rules! log_err_bail {
    ($e:expr) => {
        match $e {
            Ok(v) => v,
            Err(err) => {
                error!("{err}");
                return;
            }
        }
    };
}

#[cfg(feature = "gui")]
#[derive(Debug, Default, Clone)]
enum Message {
    #[default]
    NoOp,
    SelectFeaturesFile,
}

#[derive(Debug, Default, Clone)]
#[cfg(feature = "gui")]
pub struct Gui {
    images: Vec<PathBuf>,
    features: std::collections::HashMap<PathBuf, Faces>,
}

#[cfg(feature = "gui")]
impl iced::Sandbox for Gui {
    type Message = Message;

    fn new() -> Self {
        Self::default()
    }

    fn title(&self) -> String {
        "Extract Facial Features".to_string()
    }

    fn update(&mut self, message: Self::Message) {
        match message {
            Message::NoOp => {}
            Message::SelectFeaturesFile => {
                if let Some(file) = rfd::FileDialog::new()
                    .set_title("Open Encoded Features")
                    .pick_file()
                {
                    let data =
                        log_err_bail!(std::fs::read(&file)
                            .with_context(|| format!("reading {}", file.display())));
                    self.features =
                        log_err_bail!(ron::de::from_bytes(&data).context("decoding features"));
                }
            }
        }
    }

    fn view(&self) -> iced::Element<'_, Self::Message> {
        use iced::widget::button;
        use iced::widget::column;

        column![
            iced::widget::vertical_space(iced::Length::Fill),
            button("Open Encoded Features").on_press(Message::SelectFeaturesFile),
            iced::widget::vertical_space(iced::Length::Fill),
        ]
        .align_items(iced::Alignment::Center)
        .spacing(8)
        .into()
    }
}
